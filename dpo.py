"""
DPO(Direct Preference Optimization) 학습 스크립트.

핸드아웃 7.4가 직접 언급한 확장 — 소네트 데이터는 양성(정답)만 있으므로
음성(losing) 샘플을 직접 만들어야 한다. 본 구현은 두 종류를 모두 지원한다:

  - 휴리스틱 손상 (handout: "Heuristic Modifications"):
      shuffle / repeat / truncate / mangle / drop_punct
  - 자가생성 음성 (handout: "Automatic Generation"):
      SFT 베이스라인 모델로 높은 temperature 샘플링 → completion 생성.
      비싸므로 캐시(JSON)에 저장해 두고 재사용한다.

DPO 손실 (Rafailov et al., 2023):
  L = -E[ log σ( β * ((logπ_pol(y_w|x) - logπ_ref(y_w|x))
                    - (logπ_pol(y_l|x) - logπ_ref(y_l|x))) ) ]

핵심 구현 포인트:
  - sequence log-prob는 completion 토큰 위치에서만 합산(prompt 부분은 무시).
  - reference 모델은 SFT 체크포인트의 frozen 복사본. policy도 같은 체크포인트로 시작.
    LoRA를 정책에만 주입하면 시작점에서 정책=참조(LoRA B=0이라 delta=0) → 안정적.
  - dev CHRF로 best 추적, early stopping.

사용 예시(A100 MIG, gpt2-large LoRA):

  # (1) 베이스라인 SFT 학습 (기존 스크립트):
  python sonnet_generation.py --use_gpu --model_size gpt2-large \
         --use_lora --lora_rank 16 --grad_checkpoint \
         --batch_size 2 --epochs 8 --lr 2e-4

  # (2) (선택) 자가생성 음성 캐시 미리 만들기:
  python dpo.py --use_gpu \
         --sft_ckpt gpt2-large-lora-8ep-0.0002-sonnet.pt \
         --gen_self_negs --num_self_negs 3 --self_neg_temperature 1.4 \
         --self_neg_cache predictions/self_negs.json

  # (3) DPO 학습:
  python dpo.py --use_gpu \
         --sft_ckpt gpt2-large-lora-8ep-0.0002-sonnet.pt \
         --neg_modes shuffle,repeat,truncate,self \
         --self_neg_cache predictions/self_negs.json \
         --beta 0.1 --epochs 5 --batch_size 2 --lr 5e-5 \
         --filepath gpt2-large-lora-dpo.pt
"""

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import SonnetsDataset
from optimizer import AdamW
from sonnet_generation import (
  SonnetGPT,
  evaluate_dev_chrf,
  seed_everything,
  save_model,
)


TQDM_DISABLE = False


# ---------------------------------------------------------------------------
# 음성 샘플 휴리스틱
# ---------------------------------------------------------------------------

def _shuffle_lines(lines, rng):
  out = list(lines)
  if len(out) > 1:
    rng.shuffle(out)
  return out


def _repeat_lines(lines, rng):
  """랜덤한 한 줄을 두 번 연속 등장시키고, 다른 한 줄을 제거 → 반복 degeneracy 모사."""
  out = list(lines)
  if len(out) <= 2:
    return out
  i = rng.randrange(len(out))
  j = rng.randrange(len(out))
  if i == j:
    j = (j + 1) % len(out)
  out[j] = out[i]
  return out


def _truncate_lines(lines, rng):
  """뒷부분을 잘라 너무 짧은 완성 → 14행 형식 위반."""
  if len(lines) <= 2:
    return list(lines)
  cut = rng.randint(max(1, len(lines) // 3), max(1, len(lines) - 2))
  return list(lines[:cut])


def _mangle_lines(lines, rng):
  """각 줄의 단어 순서를 일부 뒤섞는다 — 운율/문법 둘 다 깬다."""
  out = []
  for ln in lines:
    words = ln.split()
    if len(words) <= 3:
      out.append(ln); continue
    # 인접한 두 단어를 1~2회 swap.
    for _ in range(rng.randint(1, 2)):
      i = rng.randrange(len(words) - 1)
      words[i], words[i + 1] = words[i + 1], words[i]
    out.append(' '.join(words))
  return out


def _drop_punct(lines, rng):
  """구두점 제거 → 셰익스피어 스타일에서 멀어지는 음성."""
  import string
  trans = str.maketrans('', '', string.punctuation)
  return [ln.translate(trans) for ln in lines]


HEURISTICS = {
  'shuffle': _shuffle_lines,
  'repeat': _repeat_lines,
  'truncate': _truncate_lines,
  'mangle': _mangle_lines,
  'drop_punct': _drop_punct,
}


def make_heuristic_negative(completion_lines, mode, rng):
  fn = HEURISTICS[mode]
  return '\n'.join(fn(completion_lines, rng))


# ---------------------------------------------------------------------------
# 자가생성 음성 캐시
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_self_negatives(model: SonnetGPT, dataset: SonnetsDataset, args, device):
  """SFT 모델로 prompt별 K개 completion을 샘플링해 캐시 dict를 만든다.

  반환 형식: { str(id): [neg_text_1, neg_text_2, ...] }
  각 neg_text는 prompt를 포함하지 않은 completion 텍스트(생성된 뒷부분).
  """
  cache = {}
  model.eval()
  for sid, full_text in tqdm(dataset, desc='self-neg gen', disable=TQDM_DISABLE):
    # 학습용 sonnets.json은 전체 text를 sonnets[i]로 반환한다(REPORT 참고).
    # 우리는 앞 3줄만 prompt로 쓰고 그 뒤를 생성하므로 prompt를 추출한다.
    lines = full_text.split('\n')
    prompt_text = '\n'.join(lines[:3])
    encoding = model.tokenizer(prompt_text, return_tensors='pt', padding=False, truncation=True).to(device)
    prompt_len_tok = encoding['input_ids'].shape[1]

    negs = []
    for _ in range(args.num_self_negs):
      tokens, _ = model.generate(
        encoding['input_ids'],
        temperature=args.self_neg_temperature,
        top_p=args.self_neg_top_p,
        max_length=args.self_neg_max_length,
      )
      gen = tokens[0, prompt_len_tok:].tolist()
      neg_text = model.tokenizer.decode(gen, skip_special_tokens=True).lstrip('\n')
      if neg_text.strip():
        negs.append(neg_text)
    cache[str(sid)] = negs

  return cache


# ---------------------------------------------------------------------------
# DPO 데이터셋
# ---------------------------------------------------------------------------

class DPODataset(Dataset):
  """(prompt, y_w, y_l) 트리플을 만들어 토크나이즈하는 컬렉터.

  학습용 sonnets.json의 각 sonnet에 대해 모드별로 N개의 음성을 생성하고,
  (prompt, true_completion, neg) 트리플을 모두 풀어 둔다.
  """

  def __init__(self, sonnets_path, tokenizer, neg_modes, self_neg_cache, max_length=384, seed=11711):
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.examples = self._build(sonnets_path, neg_modes, self_neg_cache, seed)

  def _build(self, path, neg_modes, self_neg_cache, seed):
    rng = random.Random(seed)
    triples = []
    # 학습용 JSON에서 prompt/completion 분리 필드 사용.
    with open(path, 'r', encoding='utf-8') as f:
      payload = json.load(f)
    for ex in payload['examples']:
      sid = ex.get('id')
      prompt = ex['prompt_text']
      completion_lines = ex['completion_lines']
      completion = '\n'.join(completion_lines)

      pos_text = '\n' + completion  # prompt와 자연스럽게 연결되도록 개행으로 시작
      # 휴리스틱 음성
      for mode in neg_modes:
        if mode == 'self':
          continue
        if mode not in HEURISTICS:
          raise ValueError(f"unknown neg mode: {mode}")
        neg = make_heuristic_negative(completion_lines, mode, rng)
        if neg.strip() and neg != completion:
          triples.append((prompt, pos_text, '\n' + neg))
      # 자가생성 음성
      if 'self' in neg_modes and self_neg_cache is not None:
        for neg in self_neg_cache.get(str(sid), []):
          if neg.strip():
            triples.append((prompt, pos_text, neg if neg.startswith('\n') else '\n' + neg))
    return triples

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, idx):
    return self.examples[idx]

  def collate_fn(self, batch):
    """배치 단위 토크나이즈. 두 시퀀스(prompt+y_w, prompt+y_l)를 각각 패딩.

    반환 keys:
      w_ids, w_mask, w_prompt_lens
      l_ids, l_mask, l_prompt_lens
    """
    w_seqs, l_seqs, w_plens, l_plens = [], [], [], []
    for prompt, y_w, y_l in batch:
      # prompt를 단독 토크나이즈해 길이 측정 (completion 손실 마스크용).
      p_ids = self.tokenizer(prompt, return_tensors=None, add_special_tokens=False)['input_ids']
      plen = len(p_ids)
      w_seqs.append(prompt + y_w)
      l_seqs.append(prompt + y_l)
      w_plens.append(plen)
      l_plens.append(plen)

    w_enc = self.tokenizer(w_seqs, return_tensors='pt', padding=True, truncation=True,
                           max_length=self.max_length)
    l_enc = self.tokenizer(l_seqs, return_tensors='pt', padding=True, truncation=True,
                           max_length=self.max_length)
    return {
      'w_ids': w_enc['input_ids'],
      'w_mask': w_enc['attention_mask'],
      'w_prompt_lens': torch.tensor(w_plens, dtype=torch.long),
      'l_ids': l_enc['input_ids'],
      'l_mask': l_enc['attention_mask'],
      'l_prompt_lens': torch.tensor(l_plens, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# 시퀀스 log-prob 계산
# ---------------------------------------------------------------------------

def sequence_logprobs(model: SonnetGPT, input_ids, attention_mask, prompt_lens):
  """각 예제에 대해 completion 토큰만 합한 log P(y|x)를 반환 (shape [B]).

  - logits[:, :-1] 가 다음 토큰을 예측 (시점 t의 logit ↔ 라벨 t+1).
  - 라벨 위치(t+1)가 (a) padding이 아니고 (b) prompt 영역 너머일 때만 손실 집계.
  """
  logits = model(input_ids, attention_mask)  # [B, T, V]
  shift_logits = logits[:, :-1, :]
  shift_labels = input_ids[:, 1:]
  shift_mask = attention_mask[:, 1:].to(shift_logits.dtype)

  B, Tm1, V = shift_logits.shape
  # completion 마스크: 라벨 위치 인덱스(1..T-1)가 prompt_len 이상인 곳만 1.
  positions = torch.arange(1, Tm1 + 1, device=input_ids.device).unsqueeze(0).expand(B, -1)
  comp_mask = (positions >= prompt_lens.unsqueeze(1)).to(shift_logits.dtype) * shift_mask

  log_probs = F.log_softmax(shift_logits, dim=-1)
  token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
  return (token_logp * comp_mask).sum(dim=1)  # [B]


# ---------------------------------------------------------------------------
# 모델 로드 (SFT → policy + ref)
# ---------------------------------------------------------------------------

def _strip_lora_from_state(state):
  """과거 LoRA 학습 ckpt를 풀-FT 구조에 로드해야 할 때, lora 키와 base 접두어를 정리.

  여기서는 SFT가 LoRA였든 풀 FT였든 그대로 SonnetGPT에 동일 args로 복원하므로 보통 불필요.
  """
  return state


def build_policy_and_reference(args, device):
  """SFT 체크포인트를 로드해 policy / reference 두 모델을 만든다.

  - 두 모델 모두 saved['args']로 동일 아키텍처 복원 후 가중치 로드.
  - DPO 추가 LoRA가 켜지면, policy에만 추가로 LoRA를 주입하여 학습 대상으로 삼는다.
    (SFT가 이미 LoRA였다면 동일한 base 위에 또 LoRA를 얹는 건 비추천. 그 경우엔
     --extra_lora 를 끄고 SFT LoRA 파라미터를 그대로 학습한다.)
  """
  print(f"[dpo] loading SFT checkpoint: {args.sft_ckpt}")
  saved = torch.load(args.sft_ckpt, map_location='cpu', weights_only=False)
  sft_args = saved['args']

  # init_only=True: HF gpt2-large 가중치 다운로드/로드 스킵 (cgroup 9GB 환경 OOM 회피).
  # state_dict로 어차피 덮어쓰므로 빈 모델로 초기화하면 된다.
  import gc

  # policy
  policy = SonnetGPT(sft_args, init_only=True)
  policy.load_state_dict(saved['model'])
  policy = policy.to(device)
  gc.collect()
  torch.cuda.empty_cache()

  # reference: 정확히 같은 구조의 frozen 복사본.
  reference = SonnetGPT(sft_args, init_only=True)
  reference.load_state_dict(saved['model'])
  reference = reference.to(device)
  del saved
  gc.collect()
  torch.cuda.empty_cache()
  for p in reference.parameters():
    p.requires_grad = False
  reference.eval()

  # policy 측 학습 대상 설정.
  if args.extra_lora:
    # SFT 위에 새 LoRA를 한 겹 더 얹는다(보통 SFT가 풀-FT일 때만 권장).
    from modules.lora import inject_lora, mark_only_lora_as_trainable
    n = inject_lora(policy.gpt, rank=args.dpo_lora_rank, alpha=args.dpo_lora_alpha,
                    dropout=args.dpo_lora_dropout)
    mark_only_lora_as_trainable(policy.gpt)
    policy.use_lora = True
    print(f"[dpo] extra LoRA injected ({n} layers, rank={args.dpo_lora_rank})")
  elif getattr(sft_args, 'use_lora', False):
    # SFT가 이미 LoRA → 그 LoRA 파라미터만 학습 대상.
    from modules.lora import mark_only_lora_as_trainable
    mark_only_lora_as_trainable(policy.gpt)
    print(f"[dpo] training existing SFT LoRA parameters")
  else:
    # 풀 FT (큰 모델엔 비추천, 작은 모델 실험용).
    for p in policy.parameters():
      p.requires_grad = True
    print(f"[dpo] full fine-tune (no LoRA)")

  trn = sum(p.numel() for p in policy.parameters() if p.requires_grad)
  tot = sum(p.numel() for p in policy.parameters())
  print(f"[dpo] policy trainable {trn:,} / {tot:,} ({100*trn/max(tot,1):.3f}%)")
  return policy, reference, sft_args


# ---------------------------------------------------------------------------
# DPO 학습 루프
# ---------------------------------------------------------------------------

def dpo_train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  policy, reference, sft_args = build_policy_and_reference(args, device)

  # SFT args를 가져오되, dev 평가용 인자(temperature/top_p/max_lines 등)는 현재 args로 override.
  # evaluate_dev_chrf 는 args.temperature, args.top_p 등을 본다.
  for fld in ('temperature', 'top_p', 'top_k', 'repetition_penalty',
              'no_repeat_ngram_size', 'max_lines'):
    if hasattr(args, fld):
      setattr(sft_args, fld, getattr(args, fld))

  # --- 자가생성 음성 캐시 처리 ---
  self_cache = None
  if 'self' in args.neg_modes:
    if args.gen_self_negs:
      # SFT 데이터로 캐시 생성 후 디스크에 저장 (재현 가능).
      sds = SonnetsDataset(args.sonnet_path)
      self_cache = generate_self_negatives(policy, sds, args, device)
      os.makedirs(os.path.dirname(args.self_neg_cache) or '.', exist_ok=True)
      with open(args.self_neg_cache, 'w', encoding='utf-8') as f:
        json.dump(self_cache, f, ensure_ascii=False, indent=2)
      print(f"[dpo] self-neg cache saved to {args.self_neg_cache} "
            f"({sum(len(v) for v in self_cache.values())} negatives total)")
    elif os.path.exists(args.self_neg_cache):
      with open(args.self_neg_cache, 'r', encoding='utf-8') as f:
        self_cache = json.load(f)
      print(f"[dpo] loaded self-neg cache from {args.self_neg_cache}")
    else:
      print(f"[dpo] WARN: --neg_modes에 'self'가 있으나 캐시 파일 없음 → self 음성 비활성.")
      args.neg_modes = [m for m in args.neg_modes if m != 'self']

  # --- DPO 데이터셋 ---
  ds = DPODataset(args.sonnet_path, policy.tokenizer, args.neg_modes, self_cache,
                  max_length=args.max_seq_length, seed=args.seed)
  print(f"[dpo] preference pairs: {len(ds)} (modes={args.neg_modes})")
  loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=ds.collate_fn)

  # --- 옵티마이저 ---
  trainable = [p for p in policy.parameters() if p.requires_grad]
  optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

  best_chrf = float('-inf')
  best_epoch = -1
  history = []
  no_improve = 0

  dev_enabled = (not args.no_dev_eval
                 and os.path.exists(args.dev_prompt_path)
                 and os.path.exists(args.dev_gold_path))

  for epoch in range(args.epochs):
    policy.train()
    epoch_loss = 0.0
    accs = 0.0
    n_batches = 0
    t0 = time.time()

    for batch in tqdm(loader, desc=f'dpo-{epoch}', disable=TQDM_DISABLE):
      w_ids = batch['w_ids'].to(device)
      w_mask = batch['w_mask'].to(device)
      l_ids = batch['l_ids'].to(device)
      l_mask = batch['l_mask'].to(device)
      w_plen = batch['w_prompt_lens'].to(device)
      l_plen = batch['l_prompt_lens'].to(device)

      # policy log-probs (그래디언트 흐름).
      pol_w = sequence_logprobs(policy, w_ids, w_mask, w_plen)
      pol_l = sequence_logprobs(policy, l_ids, l_mask, l_plen)
      # reference log-probs (no grad).
      with torch.no_grad():
        ref_w = sequence_logprobs(reference, w_ids, w_mask, w_plen)
        ref_l = sequence_logprobs(reference, l_ids, l_mask, l_plen)

      # DPO 로짓 / 손실.
      logits = args.beta * ((pol_w - ref_w) - (pol_l - ref_l))
      loss = -F.logsigmoid(logits).mean()
      # 진단용: implicit reward의 부호가 맞는 비율(>0이면 winning을 선호).
      acc = (logits > 0).float().mean().item()

      optimizer.zero_grad()
      loss.backward()
      if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
      optimizer.step()

      epoch_loss += loss.item()
      accs += acc
      n_batches += 1

    epoch_loss /= max(n_batches, 1)
    accs /= max(n_batches, 1)
    epoch_time = time.time() - t0
    rec = {'epoch': epoch, 'dpo_loss': epoch_loss, 'pref_acc': accs,
           'epoch_time_sec': epoch_time}
    print(f"[dpo] epoch {epoch}: loss={epoch_loss:.4f}  pref_acc={accs:.3f}  time={epoch_time:.1f}s")

    if dev_enabled:
      dev_chrf, _ = evaluate_dev_chrf(
        policy, sft_args, device,
        prompt_path=args.dev_prompt_path,
        gold_path=args.dev_gold_path,
        max_length=args.dev_max_new_tokens,
        verbose=args.dev_verbose,
      )
      rec['dev_chrf'] = dev_chrf
      improved = dev_chrf > best_chrf
      print(f"[dpo] epoch {epoch}: dev_chrf={dev_chrf:.4f} "
            f"(best={max(best_chrf, dev_chrf):.4f})")
      if improved:
        best_chrf = dev_chrf
        best_epoch = epoch
        no_improve = 0
        save_model(policy, optimizer, sft_args, args.filepath)
      else:
        no_improve += 1
        if args.patience > 0 and no_improve >= args.patience:
          history.append(rec)
          print(f"[dpo] early-stop at epoch {epoch} (no improve {no_improve} epochs)")
          break
    else:
      save_model(policy, optimizer, sft_args, args.filepath)

    history.append(rec)

  if args.log_path:
    os.makedirs(os.path.dirname(args.log_path) or '.', exist_ok=True)
    with open(args.log_path, 'w', encoding='utf-8') as f:
      json.dump({
        'best_epoch': best_epoch,
        'best_dev_chrf': best_chrf if best_chrf != float('-inf') else None,
        'history': history,
        'config': {
          'sft_ckpt': args.sft_ckpt,
          'beta': args.beta,
          'lr': args.lr,
          'epochs': args.epochs,
          'batch_size': args.batch_size,
          'weight_decay': args.weight_decay,
          'neg_modes': args.neg_modes,
          'extra_lora': args.extra_lora,
          'dpo_lora_rank': args.dpo_lora_rank if args.extra_lora else None,
        },
      }, f, ensure_ascii=False, indent=2)


def get_args():
  p = argparse.ArgumentParser()
  # 입력 데이터
  p.add_argument("--sonnet_path", type=str, default="data/json/sonnets.json")
  p.add_argument("--dev_prompt_path", type=str, default="data/json/sonnets_held_out_dev.json")
  p.add_argument("--dev_gold_path", type=str, default="data/json/TRUE_sonnets_held_out_dev.json")
  p.add_argument("--no_dev_eval", action='store_true')
  p.add_argument("--dev_max_new_tokens", type=int, default=128)
  p.add_argument("--dev_verbose", action='store_true')

  # SFT 체크포인트
  p.add_argument("--sft_ckpt", type=str, required=True,
                 help="sonnet_generation.py 가 저장한 SFT 체크포인트 경로.")
  p.add_argument("--filepath", type=str, default="dpo-best.pt",
                 help="DPO best 체크포인트 저장 경로.")
  p.add_argument("--log_path", type=str, default="predictions/dpo_train_log.json")

  # 학습 하이퍼파라미터
  p.add_argument("--epochs", type=int, default=5)
  p.add_argument("--batch_size", type=int, default=4)
  p.add_argument("--lr", type=float, default=5e-5)
  p.add_argument("--weight_decay", type=float, default=0.0)
  p.add_argument("--beta", type=float, default=0.1, help="DPO temperature β.")
  p.add_argument("--grad_clip", type=float, default=1.0)
  p.add_argument("--max_seq_length", type=int, default=384)
  p.add_argument("--patience", type=int, default=2)
  p.add_argument("--seed", type=int, default=11711)
  p.add_argument("--use_gpu", action='store_true')

  # DPO 위에 새 LoRA 얹기 (SFT가 풀-FT일 때만 권장).
  p.add_argument("--extra_lora", action='store_true')
  p.add_argument("--dpo_lora_rank", type=int, default=16)
  p.add_argument("--dpo_lora_alpha", type=int, default=32)
  p.add_argument("--dpo_lora_dropout", type=float, default=0.05)

  # 음성 샘플
  p.add_argument("--neg_modes", type=str, default="shuffle,repeat,truncate",
                 help="콤마구분: shuffle,repeat,truncate,mangle,drop_punct,self")
  p.add_argument("--self_neg_cache", type=str, default="predictions/self_negs.json")
  p.add_argument("--gen_self_negs", action='store_true',
                 help="자가생성 음성을 새로 만들어 캐시에 저장한다(시간이 걸린다).")
  p.add_argument("--num_self_negs", type=int, default=3)
  p.add_argument("--self_neg_temperature", type=float, default=1.4)
  p.add_argument("--self_neg_top_p", type=float, default=0.95)
  p.add_argument("--self_neg_max_length", type=int, default=160)

  # 디코딩 (dev CHRF 측정 시 사용)
  p.add_argument("--temperature", type=float, default=1.2)
  p.add_argument("--top_p", type=float, default=0.9)
  p.add_argument("--top_k", type=int, default=0)
  p.add_argument("--repetition_penalty", type=float, default=1.0)
  p.add_argument("--no_repeat_ngram_size", type=int, default=0)
  p.add_argument("--max_lines", type=int, default=0)

  args = p.parse_args()
  args.neg_modes = [m.strip() for m in args.neg_modes.split(',') if m.strip()]
  return args


if __name__ == "__main__":
  args = get_args()
  seed_everything(args.seed)
  dpo_train(args)
