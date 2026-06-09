'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import json
import os
import random
import time
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange
from sacrebleu.metrics import CHRF

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model
from modules.lora import (
  inject_lora,
  mark_only_lora_as_trainable,
  count_trainable_parameters,
  trainable_parameters,
  DEFAULT_LORA_TARGETS,
)

from optimizer import AdamW

TQDM_DISABLE = False


def _apply_repetition_penalty(logits, token_ids, penalty):
  """CTRL식 repetition penalty (Keskar et al.). 이미 등장한 토큰의 logit을
  penalty로 나눈다(양수면 작게, 음수면 더 작게). logits: [1, V], token_ids: [1, T]."""
  prev = token_ids[0].tolist()
  if not prev:
    return logits
  idx = torch.tensor(sorted(set(prev)), device=logits.device, dtype=torch.long)
  scores = logits[0, idx]
  scores = torch.where(scores > 0, scores / penalty, scores * penalty)
  logits[0, idx] = scores
  return logits


def _shift_for_lm(logits, labels):
  """Next-token CE를 위해 logits를 [:-1], labels를 [1:]로 시프트."""
  return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()


def _ce_lm_loss(logits, labels, mask=None):
  """[B, T-1, V] logits + [B, T-1] labels로 토큰 단위 mean CE.

  mask가 주어지면 (예: attention_mask[:, 1:]) padding 토큰을 평균에서 제외한다.
  GPT-2는 pad_token=eos_token이라 ignore_index로는 구분 불가 → 명시적 mask 필수.
  """
  V = logits.size(-1)
  ce = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction='none')
  if mask is None:
    return ce.mean()
  m = mask.reshape(-1).to(ce.dtype)
  return (ce * m).sum() / m.sum().clamp(min=1)


def _symmetric_kl(logits_p, logits_q):
  """대칭 KL: ½(KL(p‖q) + KL(q‖p)).

  logits: [B, T-1, V]. token-level mean. log_softmax 두 번 → numerically stable."""
  log_p = F.log_softmax(logits_p, dim=-1)
  log_q = F.log_softmax(logits_q, dim=-1)
  p, q = log_p.exp(), log_q.exp()
  kl_pq = (p * (log_p - log_q)).sum(-1).mean()
  kl_qp = (q * (log_q - log_p)).sum(-1).mean()
  return 0.5 * (kl_pq + kl_qp)


def _banned_ngram_tokens(prev_tokens, n):
  """직전 (n-1)-gram에 이어 등장하면 n-gram 반복이 되는 토큰들의 리스트를 반환."""
  if n <= 0 or len(prev_tokens) < n:
    return []
  prefix = tuple(prev_tokens[-(n - 1):]) if n > 1 else tuple()
  banned = []
  # 과거 시퀀스를 훑어 동일 prefix 뒤에 온 토큰을 모은다.
  for i in range(len(prev_tokens) - n + 1):
    if tuple(prev_tokens[i:i + n - 1]) == prefix:
      banned.append(prev_tokens[i + n - 1])
  return banned


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args, init_only=False):
    super().__init__()
    # dropout / gradient checkpointing 오버라이드. 과거(베이스라인) 체크포인트의 args 에는
    # 이 필드들이 없을 수 있으므로 getattr 기본값으로 안전하게 읽는다.
    hidden_dropout = getattr(args, 'hidden_dropout', 0.1)
    attn_dropout = getattr(args, 'attn_dropout', 0.1)
    grad_ckpt = getattr(args, 'grad_checkpoint', False)

    # init_only=True 면 HuggingFace 가중치 다운로드/로드를 건너뛴다 (ckpt에서 state_dict로
    # 덮어쓸 때 ~3GB 메모리 절감 — 9GB cgroup 제약 환경에서 OOM 회피용).
    self.gpt = GPT2Model.from_pretrained(
      model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads,
      hidden_dropout_prob=hidden_dropout,
      attention_probs_dropout_prob=attn_dropout,
      gradient_checkpointing=grad_ckpt,
      init_only=init_only,
    )
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # PEFT(LoRA) 경로: 베이스를 freeze 하고 저랭크 보정항만 학습한다.
    self.use_lora = getattr(args, 'use_lora', False)
    if self.use_lora:
      targets = getattr(args, 'lora_targets', None) or DEFAULT_LORA_TARGETS
      n = inject_lora(
        self.gpt,
        rank=getattr(args, 'lora_rank', 8),
        alpha=getattr(args, 'lora_alpha', 16),
        dropout=getattr(args, 'lora_dropout', 0.0),
        targets=targets,
      )
      mark_only_lora_as_trainable(self.gpt, train_layernorm=getattr(args, 'lora_train_ln', False))
      trn, tot = count_trainable_parameters(self.gpt)
      print(f"[lora] injected into {n} linear layers (targets={tuple(targets)}); "
            f"trainable {trn:,} / {tot:,} params ({100*trn/max(tot,1):.3f}%)")
    else:
      # 기존 동작: 전체 모델 fine-tuning.
      for param in self.gpt.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    ### YOUR CODE HERE
    raise NotImplementedError

    sequence_output = outputs['last_hidden_state']   # [B, T, H]

    # GPT-2의 token embedding weight를 LM head처럼 사용 (weight tying).
    # word_embedding은 GPT2Model에 직접 정의돼 있다 (embed는 메서드라 .word_embedding 없음).
    logits = F.linear(sequence_output, self.gpt.word_embedding.weight)  # [B, T, V]

    return logits

  def forward_from_embeds(self, embeds, attention_mask):
    """임베딩에서 시작하는 forward (SMART의 perturbed forward 용).

    GPT2Model.embed → encode → final_layer_norm → LM head(=weight tying)의 후반부.
    SMART는 embed 결과에 δ를 더해 다시 흘려야 하므로 이 진입점이 필요하다.
    """
    hidden = self.gpt.encode(embeds, attention_mask)
    hidden = self.gpt.final_layer_norm(hidden)
    logits = F.linear(hidden, self.gpt.word_embedding.weight)
    return logits

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128,
               top_k=0, repetition_penalty=1.0, no_repeat_ngram_size=0, max_lines=0):
    """
    Top-p(nucleus) + temperature sampling 기반 소네트 생성기.

    기본 인자는 기존 베이스라인 동작을 그대로 재현한다(top_k=0, repetition_penalty=1.0,
    no_repeat_ngram_size=0, max_lines=0 → 모두 비활성). 아래 옵션을 켜면 디코딩 품질을
    개선할 수 있다(학습 무관, 즉시 적용):

      - top_k:               확률 상위 k개 토큰만 후보로 (0이면 비활성).
      - repetition_penalty:  이미 등장한 토큰의 logit을 penalty로 나눠 반복을 억제 (>1.0 권장 1.1~1.3).
      - no_repeat_ngram_size: 동일 n-gram 재등장 금지 (소네트의 통째 행 반복 방지에 효과적).
      - max_lines:           생성 텍스트가 이 줄 수(개행 기준)에 도달하면 종료. 소네트는 14.
    """
    device = self.get_device()
    token_ids = encoding.to(device)
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64, device=device)

    # 줄 수 기반 종료를 위해 프롬프트가 이미 가진 개행 수를 센다.
    line_count = self.tokenizer.decode(token_ids[0].tolist()).count('\n') if max_lines > 0 else 0

    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :]  # [1, V]

      # (1) Repetition penalty: 이미 생성된 토큰의 logit을 조정.
      if repetition_penalty != 1.0:
        logits_last_token = _apply_repetition_penalty(logits_last_token, token_ids, repetition_penalty)

      # (2) no_repeat_ngram: 직전 (n-1)-gram을 반복 완성할 토큰을 차단.
      if no_repeat_ngram_size > 0:
        banned = _banned_ngram_tokens(token_ids[0].tolist(), no_repeat_ngram_size)
        if banned:
          logits_last_token[0, banned] = float('-inf')

      # Apply temperature scaling
      logits_last_token = logits_last_token / temperature

      # (3) Top-k 필터링.
      if top_k and top_k > 0:
        k = min(top_k, logits_last_token.size(-1))
        kth_value = torch.topk(logits_last_token, k, dim=-1).values[..., -1, None]
        logits_last_token = logits_last_token.masked_fill(logits_last_token < kth_value, float('-inf'))

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # (4) Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64, device=device)], dim=1
      )

      # (5) 줄 수 기반 조기 종료: 14행 소네트 형식 유지.
      if max_lines > 0:
        line_count += self.tokenizer.decode([sampled_token.item()]).count('\n')
        if line_count >= max_lines:
          break

    # NOTE: starter code 의 [3:] 슬라이스는 GPT-2엔 BOS가 없어 prompt 첫 3 글자를 그냥 잘라내는
    # 버그였다 ("From..." → "m..."). 제거하여 prompt가 온전히 보존되도록 한다.
    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


@torch.no_grad()
def evaluate_dev_chrf(model, args, device,
                     prompt_path, gold_path,
                     max_length=128, verbose=False):
  """Run generation on the dev prompt set and score it against the gold sonnets
  with sacreBLEU CHRF. Returns (chrf_score, generated_sonnets).

  generated_sonnets is a list of full decoded strings (prompt + completion).
  """
  model.eval()
  prompt_ds = SonnetsDataset(prompt_path)
  gold_ds = SonnetsDataset(gold_path)
  assert len(prompt_ds) == len(gold_ds), (
    f"dev prompt/gold size mismatch: {len(prompt_ds)} vs {len(gold_ds)}")

  hyps, refs = [], []
  for (pidx, prompt_text), (gidx, gold_text) in zip(prompt_ds, gold_ds):
    encoding = model.tokenizer(prompt_text, return_tensors='pt', padding=False,
                               truncation=True).to(device)
    _, decoded = model.generate(encoding['input_ids'],
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_length=max_length,
                                top_k=getattr(args, 'top_k', 0),
                                repetition_penalty=getattr(args, 'repetition_penalty', 1.0),
                                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                                max_lines=getattr(args, 'max_lines', 0))
    hyps.append(decoded)
    refs.append(gold_text)
    if verbose:
      print(f'--- dev sonnet {pidx} ---\n{decoded}\n')

  chrf_score = float(CHRF().corpus_score(hyps, [refs]).score)
  return chrf_score, hyps


def _compute_step_loss(model, b_ids, b_mask, args):
  """배치 1개에 대한 학습 손실 계산. 분기:
    - baseline:  CE(logits, labels)
    - +R-Drop:   ½(CE1+CE2) + λ·½(KL(p1‖p2)+KL(p2‖p1)),  forward 2회 (dropout 두 sampling)
    - +SMART:    CE(p) + λ·SymKL(p, p_pert),  p_pert: embed에 1-step grad-ascent로 찾은 δ
    - +both:     R-Drop의 (ce1, p1, p2)를 그대로 쓰면서, p1과 perturbed p_pert 사이에도 SymKL 추가.

  반환: (loss_tensor, dict({'ce': float, 'rdrop_kl': float|None, 'smart_kl': float|None}))
  """
  metrics = {'ce': None, 'rdrop_kl': None, 'smart_kl': None}
  use_rdrop = getattr(args, 'use_rdrop', False)
  use_smart = getattr(args, 'use_smart', False)

  # padding 토큰 마스크 (shifted): labels[t]가 padding이면 loss에서 제외.
  # pad_token = eos_token이므로 ignore_index로는 분리 불가 → attention_mask로 마스킹.
  shift_mask = b_mask[:, 1:].contiguous()

  # === 첫 번째 forward (clean) — 항상 한다.
  embeds_1 = model.gpt.embed(b_ids)
  logits_1 = model.forward_from_embeds(embeds_1, b_mask)
  shift_logits_1, labels = _shift_for_lm(logits_1, b_ids)
  ce_1 = _ce_lm_loss(shift_logits_1, labels, mask=shift_mask)
  total = ce_1

  if use_rdrop:
    # 같은 input을 다시 흘리면 embed dropout / attention dropout이 새로 sampling됨.
    embeds_2 = model.gpt.embed(b_ids)
    logits_2 = model.forward_from_embeds(embeds_2, b_mask)
    shift_logits_2, _ = _shift_for_lm(logits_2, b_ids)
    ce_2 = _ce_lm_loss(shift_logits_2, labels, mask=shift_mask)
    rdrop_kl = _symmetric_kl(shift_logits_1, shift_logits_2)
    ce_avg = 0.5 * (ce_1 + ce_2)
    total = ce_avg + args.rdrop_lambda * rdrop_kl
    metrics['ce'] = ce_avg.item()
    metrics['rdrop_kl'] = rdrop_kl.item()
  else:
    metrics['ce'] = ce_1.item()

  if use_smart:
    # 적대적 δ 초기화: gaussian → 단위 정규화 → ε로 스케일.
    delta = torch.randn_like(embeds_1) * args.smart_eps
    delta = delta.detach().requires_grad_(True)

    # detach된 clean logits (KL 타겟). grad 추적 비활성화로 메모리/계산 절약.
    target_logits = shift_logits_1.detach()

    # δ를 SMART_STEPS 만큼 grad ascent (KL을 키우는 방향)로 업데이트.
    for _ in range(max(1, args.smart_steps)):
      pert_logits = model.forward_from_embeds(embeds_1 + delta, b_mask)
      shift_pert, _ = _shift_for_lm(pert_logits, b_ids)
      adv_kl = _symmetric_kl(shift_pert, target_logits)
      delta_grad = torch.autograd.grad(adv_kl, delta, retain_graph=False, create_graph=False)[0]
      # token-position마다 norm 정규화 → α 만큼 한 발 전진.
      gnorm = delta_grad.norm(dim=-1, keepdim=True) + 1e-8
      with torch.no_grad():
        delta_new = delta + args.smart_alpha * delta_grad / gnorm
        # ε-ball로 투영 (per-token L2).
        dnorm = delta_new.norm(dim=-1, keepdim=True)
        scale = torch.clamp(args.smart_eps / (dnorm + 1e-8), max=1.0)
        delta_new = delta_new * scale
      delta = delta_new.detach().requires_grad_(True)

    # 최종 perturbed forward (이번엔 model 파라미터로 backprop 흐름).
    pert_logits = model.forward_from_embeds(embeds_1 + delta.detach(), b_mask)
    shift_pert, _ = _shift_for_lm(pert_logits, b_ids)
    smart_kl = _symmetric_kl(shift_logits_1, shift_pert)
    total = total + args.smart_lambda * smart_kl
    metrics['smart_kl'] = smart_kl.item()

  return total, metrics


def train(args):
  """Fine-tune GPT-2 for sonnet generation with dev-CHRF early stopping."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  # LoRA 사용 시엔 학습 가능한(=lora) 파라미터만 옵티마이저에 넘긴다.
  # 풀 FT면 전체 파라미터. weight_decay는 정규화 하이퍼파라미터로 노출.
  params = list(trainable_parameters(model)) if model.use_lora else list(model.parameters())
  optimizer = AdamW(params, lr=lr, weight_decay=args.weight_decay)
  trn, tot = count_trainable_parameters(model)
  print(f"[train] optimizing {trn:,} / {tot:,} params  "
        f"(lr={lr}, weight_decay={args.weight_decay}, "
        f"hidden_dropout={args.hidden_dropout}, attn_dropout={args.attn_dropout})")

  # dev evaluation 사용 가능 여부 확인. 두 파일 모두 존재해야 켜진다.
  dev_eval_enabled = (not args.no_dev_eval
                     and os.path.exists(args.dev_prompt_path)
                     and os.path.exists(args.dev_gold_path))
  if dev_eval_enabled:
    print(f"[dev] CHRF evaluation enabled: prompts={args.dev_prompt_path}, "
          f"gold={args.dev_gold_path}, patience={args.patience}")
  else:
    print('[dev] CHRF evaluation disabled (no_dev_eval flag set or files missing).')

  best_chrf = float('-inf')
  best_epoch = -1
  epochs_without_improvement = 0
  history = []

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    sum_ce = 0.0
    sum_rdrop_kl = 0.0
    sum_smart_kl = 0.0
    num_batches = 0
    t0 = time.time()

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      optimizer.zero_grad()
      loss, m = _compute_step_loss(model, b_ids, b_mask, args)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      sum_ce += m['ce']
      if m['rdrop_kl'] is not None: sum_rdrop_kl += m['rdrop_kl']
      if m['smart_kl'] is not None: sum_smart_kl += m['smart_kl']
      num_batches += 1

    nb = max(num_batches, 1)
    train_loss /= nb
    avg_ce = sum_ce / nb
    avg_rdrop_kl = (sum_rdrop_kl / nb) if args.use_rdrop else None
    avg_smart_kl = (sum_smart_kl / nb) if args.use_smart else None
    epoch_time = time.time() - t0

    aux = []
    if avg_rdrop_kl is not None: aux.append(f"rdrop_kl={avg_rdrop_kl:.4f}")
    if avg_smart_kl is not None: aux.append(f"smart_kl={avg_smart_kl:.4f}")
    aux_str = ('  ' + '  '.join(aux)) if aux else ''
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}  ce={avg_ce:.4f}{aux_str}  time={epoch_time:.1f}s")

    epoch_record = {
      'epoch': epoch,
      'train_loss': train_loss,
      'train_ce': avg_ce,
      'train_rdrop_kl': avg_rdrop_kl,
      'train_smart_kl': avg_smart_kl,
      'epoch_time_sec': epoch_time,
      'lr': lr,
    }

    if dev_eval_enabled:
      dev_chrf, _ = evaluate_dev_chrf(
        model, args, device,
        prompt_path=args.dev_prompt_path,
        gold_path=args.dev_gold_path,
        max_length=args.dev_max_new_tokens,
        verbose=args.dev_verbose,
      )
      epoch_record['dev_chrf'] = dev_chrf
      improved = dev_chrf > best_chrf
      print(f"Epoch {epoch}: dev_chrf={dev_chrf:.4f}  "
            f"(best={max(best_chrf, dev_chrf):.4f} @ epoch "
            f"{epoch if improved else best_epoch})")

      if improved:
        best_chrf = dev_chrf
        best_epoch = epoch
        epochs_without_improvement = 0
        save_model(model, optimizer, args, args.filepath)
      else:
        epochs_without_improvement += 1
        if args.patience > 0 and epochs_without_improvement >= args.patience:
          print(f"[early-stop] no dev_chrf improvement for {args.patience} epochs. "
                f"Stopping at epoch {epoch}.")
          history.append(epoch_record)
          break
    else:
      # dev 평가가 꺼져 있을 때는 매 에폭의 모델을 그대로 저장 (기존 동작 유지).
      save_model(model, optimizer, args, args.filepath)

    history.append(epoch_record)

  # 학습 로그 저장
  if args.log_path:
    os.makedirs(os.path.dirname(args.log_path) or '.', exist_ok=True)
    with open(args.log_path, 'w', encoding='utf-8') as f:
      json.dump({
        'best_epoch': best_epoch,
        'best_dev_chrf': best_chrf if best_chrf != float('-inf') else None,
        'history': history,
        'config': {
          'model_size': args.model_size,
          'epochs': args.epochs,
          'batch_size': args.batch_size,
          'lr': args.lr,
          'temperature': args.temperature,
          'top_p': args.top_p,
          'top_k': args.top_k,
          'repetition_penalty': args.repetition_penalty,
          'no_repeat_ngram_size': args.no_repeat_ngram_size,
          'max_lines': args.max_lines,
          'weight_decay': args.weight_decay,
          'hidden_dropout': args.hidden_dropout,
          'attn_dropout': args.attn_dropout,
          'grad_checkpoint': args.grad_checkpoint,
          'use_lora': args.use_lora,
          'lora_rank': args.lora_rank if args.use_lora else None,
          'lora_alpha': args.lora_alpha if args.use_lora else None,
          'lora_dropout': args.lora_dropout if args.use_lora else None,
          'lora_targets': args.lora_targets if args.use_lora else None,
          'lora_train_ln': args.lora_train_ln if args.use_lora else None,
          'patience': args.patience,
          'dev_eval_enabled': dev_eval_enabled,
          'use_rdrop': args.use_rdrop,
          'rdrop_lambda': args.rdrop_lambda if args.use_rdrop else None,
          'use_smart': args.use_smart,
          'smart_lambda': args.smart_lambda if args.use_smart else None,
          'smart_eps': args.smart_eps if args.use_smart else None,
          'smart_alpha': args.smart_alpha if args.use_smart else None,
          'smart_steps': args.smart_steps if args.use_smart else None,
        },
      }, f, ensure_ascii=False, indent=2)
    print(f"[log] training log written to {args.log_path}")

  if dev_eval_enabled and best_epoch >= 0:
    print(f"[summary] best dev_chrf={best_chrf:.4f} @ epoch {best_epoch}. "
          f"best ckpt at {args.filepath}")


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # 새 학습 루프는 args.filepath에 best checkpoint 하나만 저장한다.
  # 과거 형식(`{epoch}_{filepath}`)이 남아 있으면 그것도 fallback으로 시도.
  ckpt_path = args.filepath
  if not os.path.exists(ckpt_path):
    legacy = f'{args.epochs-1}_{args.filepath}'
    if os.path.exists(legacy):
      ckpt_path = legacy
    else:
      raise FileNotFoundError(
        f"No checkpoint found at {args.filepath} (or legacy {legacy}). "
        f"Run training first."
      )
  print(f"[submit] loading checkpoint from {ckpt_path}")
  saved = torch.load(ckpt_path, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p,
                            max_length=getattr(args, 'dev_max_new_tokens', 160),
                            top_k=getattr(args, 'top_k', 0),
                            repetition_penalty=getattr(args, 'repetition_penalty', 1.0),
                            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                            max_lines=getattr(args, 'max_lines', 0))[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  # 입력은 동료가 만든 전처리 JSON(data/json/*.json)을 기본값으로 사용한다.
  # SonnetsDataset이 .txt/.json 둘 다 읽으므로, 원본 txt 경로를 넘겨도 동작한다.
  parser.add_argument("--sonnet_path", type=str, default="data/json/sonnets.json")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/json/sonnets_held_out.json")
  # 출력 파일이 None이면 학습 구성(tag) 기반으로 자동 결정 → 다른 학습 결과를 덮어쓰지 않음.
  parser.add_argument("--sonnet_out", type=str, default=None,
                      help="held-out 생성 텍스트 출력 경로. None이면 predictions/generated_sonnets_{tag}.txt.")

  # Dev evaluation 관련.
  parser.add_argument("--dev_prompt_path", type=str, default="data/json/sonnets_held_out_dev.json",
                      help="Dev prompts (앞 3줄만). 매 epoch CHRF 평가용.")
  parser.add_argument("--dev_gold_path", type=str, default="data/json/TRUE_sonnets_held_out_dev.json",
                      help="Dev gold sonnets. 매 epoch CHRF 평가용.")
  parser.add_argument("--no_dev_eval", action='store_true',
                      help="Dev CHRF 평가 비활성화. 매 epoch 모델을 그대로 저장하는 기존 방식으로 회귀.")
  parser.add_argument("--patience", type=int, default=3,
                      help="Dev CHRF가 개선되지 않은 epoch 수가 이 값 이상이면 early stop. 0이면 비활성.")
  parser.add_argument("--dev_max_new_tokens", type=int, default=128,
                      help="Dev 평가 시 생성할 최대 토큰 수.")
  parser.add_argument("--dev_verbose", action='store_true',
                      help="Dev epoch마다 생성된 소네트를 콘솔에 출력.")
  parser.add_argument("--log_path", type=str, default=None,
                      help="에폭별 학습/평가 기록 JSON 경로. None이면 predictions/train_{tag}.json, "
                           "빈 문자열이면 저장하지 않음.")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--top_k", type=int, default=0,
                      help="Top-k 필터링 (0이면 비활성).")
  parser.add_argument("--repetition_penalty", type=float, default=1.0,
                      help="반복 억제 penalty (>1.0, 예: 1.2). 1.0이면 비활성.")
  parser.add_argument("--no_repeat_ngram_size", type=int, default=0,
                      help="동일 n-gram 반복 금지 크기 (0이면 비활성). 행 통째 반복 방지에 효과적.")
  parser.add_argument("--max_lines", type=int, default=0,
                      help="생성 텍스트가 이 줄 수에 도달하면 종료. 소네트=14 권장. 0이면 비활성.")

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  # --- 정규화 / 하이퍼파라미터 ---
  parser.add_argument("--weight_decay", type=float, default=0.0,
                      help="AdamW weight decay (정규화). 작은 데이터 과적합 방어에 0.01~0.1 시도.")
  parser.add_argument("--hidden_dropout", type=float, default=0.1,
                      help="hidden/embedding/residual dropout 확률.")
  parser.add_argument("--attn_dropout", type=float, default=0.1,
                      help="attention probability dropout 확률.")
  parser.add_argument("--grad_checkpoint", action='store_true',
                      help="gradient checkpointing 활성화 (gpt2-large 등 큰 모델을 좁은 VRAM에 적재).")

  # --- Consistency regularization (R-Drop / SMART) ---
  # 작은 데이터(131편 sonnet)에서 generalization을 개선하기 위한 두 정규화 기법.
  # 둘 다 동일 입력에 대해 두 forward 결과의 분포 일치를 강제 → consistency regularizer.
  parser.add_argument("--use_rdrop", action='store_true',
                      help="R-Drop (Liang et al. 2021). 같은 입력 두 번 forward + symmetric KL.")
  parser.add_argument("--rdrop_lambda", type=float, default=1.0,
                      help="R-Drop KL 항 weighting. NMT/분류 기본 1.0~5.0.")
  parser.add_argument("--use_smart", action='store_true',
                      help="SMART (Jiang et al. 2020). embedding adversarial perturb + symmetric KL.")
  parser.add_argument("--smart_lambda", type=float, default=1.0,
                      help="SMART KL 항 weighting.")
  parser.add_argument("--smart_eps", type=float, default=1e-5,
                      help="perturbation ε (per-token L2 ball 반지름). 임베딩 norm 대비 작게.")
  parser.add_argument("--smart_alpha", type=float, default=1e-3,
                      help="δ grad-ascent step size.")
  parser.add_argument("--smart_steps", type=int, default=1,
                      help="δ 갱신 반복 횟수 (논문 기본 1).")

  # --- PEFT (LoRA) ---
  parser.add_argument("--use_lora", action='store_true',
                      help="LoRA 파라미터 효율 fine-tuning 사용 (베이스 freeze).")
  parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank r.")
  parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (scaling = alpha/r).")
  parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA 경로 dropout.")
  parser.add_argument("--lora_targets", type=str, default=None,
                      help="LoRA 주입 대상 Linear 이름 (콤마구분). 미지정 시 q,k,v,proj,mlp 전체.")
  parser.add_argument("--lora_train_ln", action='store_true',
                      help="LoRA와 함께 LayerNorm 파라미터도 학습.")

  args = parser.parse_args()
  if args.lora_targets is not None:
    args.lora_targets = tuple(t.strip() for t in args.lora_targets.split(',') if t.strip())
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  # 실험 구성이 파일명에 드러나도록 태깅. ckpt/log/생성텍스트 셋이 같은 tag로 자동 묶임 → 덮어쓰기 방지.
  # seed/wd/dropout이 기본값과 다르면 tag에 함께 노출해 같은 (model,lr,epoch) 안에서도 구분 가능.
  method = '-lora-r' + str(args.lora_rank) if args.use_lora else '-fullft'
  tag_parts = [args.model_size + method, f"s{args.seed}", f"{args.epochs}ep", str(args.lr)]
  if args.weight_decay and args.weight_decay > 0:
    tag_parts.append(f"wd{args.weight_decay}")
  if abs(args.hidden_dropout - 0.1) > 1e-9 or abs(args.attn_dropout - 0.1) > 1e-9:
    tag_parts.append(f"hd{args.hidden_dropout}-ad{args.attn_dropout}")
  # consistency regularizer 표식 — 항상 켜진 것만 tag에 노출.
  if args.use_rdrop:
    tag_parts.append(f"rdrop{args.rdrop_lambda}")
  if args.use_smart:
    tag_parts.append(f"smart{args.smart_lambda}-e{args.smart_eps}-a{args.smart_alpha}-s{args.smart_steps}")
  tag = "-".join(tag_parts)
  # ckpt는 predictions/ckpts/ 하위로 모은다 (저장소 루트에 .pt 파일이 쌓이지 않게).
  os.makedirs('predictions/ckpts', exist_ok=True)
  args.filepath = f'predictions/ckpts/{tag}-sonnet.pt'
  if args.log_path is None:
    args.log_path = f'predictions/train_{tag}.json'
  if args.sonnet_out is None:
    args.sonnet_out = f'predictions/generated_{tag}.txt'
  print(f"[tag] {tag}\n      ckpt={args.filepath}\n      log={args.log_path}\n      out={args.sonnet_out}")
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)