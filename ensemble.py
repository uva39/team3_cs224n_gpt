"""
앙상블 추론 스크립트.

두 가지 전략을 지원한다:

  (A) candidate selection (기본):
      체크포인트 1개 또는 여러 개로 prompt당 K개 후보를 샘플링한 뒤,
      reference-free 소네트 품질 점수로 best를 골라 제출.
      소네트 평가가 CHRF(셰익스피어 분포와의 매칭)이므로,
      구조적 정합성(14행, 운율, 반복 회피)을 점수에 반영하면
      평균적으로 CHRF가 올라가는 경향이 있다.

  (B) prob-average ensemble:
      여러 체크포인트의 next-token 확률을 산술 평균하여 단일 시퀀스를 디코딩.
      vocab이 동일(GPT2 tokenizer)해야 함.

사용 예시:

  # 단일 ckpt + 후보 선택
  python ensemble.py --use_gpu \
         --ckpts gpt2-large-lora-8ep-0.0002-sonnet.pt \
         --mode select --num_candidates 8 --max_lines 14 \
         --sonnet_out predictions/generated_sonnets.txt

  # 다중 ckpt + 후보 선택
  python ensemble.py --use_gpu \
         --ckpts ckpt_a.pt,ckpt_b.pt,ckpt_c.pt \
         --mode select --num_candidates 4 --max_lines 14

  # 다중 ckpt + 확률 평균
  python ensemble.py --use_gpu --ckpts a.pt,b.pt --mode prob_avg --max_lines 14
"""

import argparse
import os
import re
from collections import Counter
from typing import List

import torch
import torch.nn.functional as F

from datasets import SonnetsDataset
from sonnet_generation import (
  SonnetGPT,
  seed_everything,
  _apply_repetition_penalty,
  _banned_ngram_tokens,
)


# ---------------------------------------------------------------------------
# Reference-free 소네트 품질 점수
# ---------------------------------------------------------------------------

_RHYME_SCHEME = ['A', 'B', 'A', 'B', 'C', 'D', 'C', 'D', 'E', 'F', 'E', 'F', 'G', 'G']


def _last_word(line):
  toks = re.findall(r"[A-Za-z']+", line)
  return toks[-1].lower() if toks else ''


def _rhyme_key(word, n=2):
  """매우 단순한 운율 키 — 단어 끝 n글자."""
  return word[-n:] if len(word) >= n else word


def sonnet_quality_score(text, weights):
  """완성된 sonnet 텍스트(prompt 포함)에 대한 reference-free 품질 점수.

  점수가 클수록 좋다. weights dict로 각 항목 가중치 조절.
  """
  # 비어 있거나 공백뿐이면 매우 낮은 점수.
  lines = [ln.strip() for ln in text.strip().split('\n') if ln.strip()]
  if len(lines) == 0:
    return -1e6

  # (1) 14행 형식: 14에 가까울수록 좋음.
  line_count_pen = -abs(len(lines) - 14)

  # (2) 평균 줄 길이 — 셰익스피어 소네트는 30~50자 근방. 그 범위를 넘어가면 페널티.
  avg_len = sum(len(ln) for ln in lines) / len(lines)
  len_pen = -max(0.0, abs(avg_len - 40) - 10) * 0.5

  # (3) 줄 단위 반복: 동일/거의-동일 줄 등장 페널티.
  dup_pen = -(len(lines) - len(set(lines)))

  # (4) n-gram 다양성: 4-gram unique 비율(낮으면 반복적).
  toks = re.findall(r"[A-Za-z']+", text.lower())
  if len(toks) >= 5:
    ngrams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    diversity = len(set(ngrams)) / max(len(ngrams), 1)
  else:
    diversity = 0.0

  # (5) 운율 적합도: 14행이 있을 때 ABAB CDCD EFEF GG 스킴에 따라
  #     같은 그룹 마지막 단어의 끝-2글자가 일치하면 보너스.
  rhyme_bonus = 0.0
  if len(lines) >= 14:
    last_words = [_last_word(ln) for ln in lines[:14]]
    keys = [_rhyme_key(w) for w in last_words]
    groups = {}
    for k, ln_idx in zip(_RHYME_SCHEME, range(14)):
      groups.setdefault(k, []).append(keys[ln_idx])
    for k, ks in groups.items():
      # 같은 그룹 내 모든 페어가 같은 끝-2글자면 +1.
      if len(set(ks)) == 1 and ks[0]:
        rhyme_bonus += 1.0
      else:
        # 부분 일치 비율.
        pairs = [(a, b) for i, a in enumerate(ks) for b in ks[i + 1:]]
        if pairs:
          rhyme_bonus += sum(1 for a, b in pairs if a and a == b) / len(pairs)

  # (6) 단조 토큰 반복 페널티: 가장 빈번한 단어가 전체의 X% 이상이면 페널티.
  if toks:
    most_common = Counter(toks).most_common(1)[0][1]
    mc_ratio = most_common / len(toks)
    mc_pen = -max(0.0, mc_ratio - 0.08) * 5.0
  else:
    mc_pen = -1.0

  w = weights
  return (
    w['line_count'] * line_count_pen
    + w['avg_len'] * len_pen
    + w['duplicate'] * dup_pen
    + w['diversity'] * diversity
    + w['rhyme'] * rhyme_bonus
    + w['mono'] * mc_pen
  )


DEFAULT_WEIGHTS = {
  'line_count': 3.0,
  'avg_len': 0.5,
  'duplicate': 2.0,
  'diversity': 1.0,
  'rhyme': 1.0,
  'mono': 1.0,
}


# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------

def load_models(paths, device):
  """체크포인트 리스트에서 SonnetGPT 모델들을 로드해 list로 반환."""
  import gc
  models = []
  tok = None
  for p in paths:
    print(f"[ensemble] loading {p}")
    saved = torch.load(p, map_location='cpu', weights_only=False)
    # init_only=True: HF 가중치 스킵 (9GB cgroup OOM 회피).
    m = SonnetGPT(saved['args'], init_only=True)
    m.load_state_dict(saved['model'])
    del saved
    gc.collect()
    m = m.to(device).eval()
    torch.cuda.empty_cache()
    if tok is None:
      tok = m.tokenizer
    models.append(m)
  return models, tok


# ---------------------------------------------------------------------------
# 후보 생성 + 선택
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_candidates(model: SonnetGPT, encoding, args):
  """동일 prompt로 N개의 후보를 샘플링한다(생성마다 RNG 상태가 바뀌어 다양해진다)."""
  cands = []
  for _ in range(args.num_candidates):
    _, decoded = model.generate(
      encoding,
      temperature=args.temperature,
      top_p=args.top_p,
      max_length=args.max_length,
      top_k=args.top_k,
      repetition_penalty=args.repetition_penalty,
      no_repeat_ngram_size=args.no_repeat_ngram_size,
      max_lines=args.max_lines,
    )
    cands.append(decoded)
  return cands


def pick_best(candidates, weights):
  """품질 점수가 가장 높은 후보를 (text, score) 로 반환."""
  scored = [(c, sonnet_quality_score(c, weights)) for c in candidates]
  scored.sort(key=lambda x: x[1], reverse=True)
  return scored[0]


# ---------------------------------------------------------------------------
# Prob-avg 앙상블 디코딩
# ---------------------------------------------------------------------------

@torch.no_grad()
def prob_avg_decode(models: List[SonnetGPT], encoding, args, device):
  """next-token 확률을 모델들 사이에서 평균낸 뒤 top-p로 샘플링."""
  token_ids = encoding.to(device)
  attention_mask = torch.ones(token_ids.shape, dtype=torch.int64, device=device)
  tok = models[0].tokenizer

  line_count = tok.decode(token_ids[0].tolist()).count('\n') if args.max_lines > 0 else 0

  for _ in range(args.max_length):
    avg_probs = None
    for m in models:
      logits = m(token_ids, attention_mask)[:, -1, :]
      if args.repetition_penalty != 1.0:
        logits = _apply_repetition_penalty(logits, token_ids, args.repetition_penalty)
      if args.no_repeat_ngram_size > 0:
        banned = _banned_ngram_tokens(token_ids[0].tolist(), args.no_repeat_ngram_size)
        if banned:
          logits[0, banned] = float('-inf')
      logits = logits / args.temperature
      if args.top_k and args.top_k > 0:
        k = min(args.top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float('-inf'))
      probs = F.softmax(logits, dim=-1)
      avg_probs = probs if avg_probs is None else (avg_probs + probs)
    avg_probs = avg_probs / len(models)

    # top-p
    sorted_probs, sorted_idx = torch.sort(avg_probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum <= args.top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = True
    filt = sorted_probs * mask
    filt = filt / filt.sum(dim=-1, keepdim=True)
    sampled = sorted_idx.gather(-1, torch.multinomial(filt, 1))

    if sampled.item() == tok.eos_token_id:
      break

    token_ids = torch.cat([token_ids, sampled], dim=1)
    attention_mask = torch.cat(
      [attention_mask, torch.ones((1, 1), dtype=torch.int64, device=device)], dim=1
    )

    if args.max_lines > 0:
      line_count += tok.decode([sampled.item()]).count('\n')
      if line_count >= args.max_lines:
        break

  return token_ids, tok.decode(token_ids[0].cpu().tolist())


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def run(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  ckpts = [c.strip() for c in args.ckpts.split(',') if c.strip()]
  models, tok = load_models(ckpts, device)
  weights = dict(DEFAULT_WEIGHTS)
  if args.weights_json and os.path.exists(args.weights_json):
    import json
    with open(args.weights_json, 'r') as f:
      weights.update(json.load(f))

  ds = SonnetsDataset(args.held_out_sonnet_path)
  out = []
  for sid, prompt in ds:
    encoding = tok(prompt, return_tensors='pt', padding=False, truncation=True).to(device)
    if args.mode == 'prob_avg':
      _, decoded = prob_avg_decode(models, encoding['input_ids'], args, device)
      best_score = sonnet_quality_score(decoded, weights)
      print(f"[id {sid}] prob_avg score={best_score:.3f}")
      out.append((sid, decoded))
    else:  # select
      # 후보 생성은 첫 모델 또는 다중 모델 round-robin.
      all_cands = []
      for m in models:
        all_cands.extend(generate_candidates(m, encoding['input_ids'], args))
      best, best_score = pick_best(all_cands, weights)
      print(f"[id {sid}] best of {len(all_cands)} cands, score={best_score:.3f}")
      out.append((sid, best))

  os.makedirs(os.path.dirname(args.sonnet_out) or '.', exist_ok=True)
  with open(args.sonnet_out, 'w', encoding='utf-8') as f:
    f.write("--Generated Sonnets-- \n\n")
    for sid, text in out:
      f.write(f"\n{sid}\n{text}\n\n")
  print(f"[ensemble] wrote {len(out)} sonnets → {args.sonnet_out}")


def get_args():
  p = argparse.ArgumentParser()
  p.add_argument("--ckpts", type=str, required=True, help="콤마구분 체크포인트 경로 리스트.")
  p.add_argument("--mode", type=str, choices=['select', 'prob_avg'], default='select')
  p.add_argument("--num_candidates", type=int, default=8,
                 help="select 모드에서 ckpt당 생성할 후보 수.")
  p.add_argument("--held_out_sonnet_path", type=str, default="data/json/sonnets_held_out.json")
  p.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
  p.add_argument("--weights_json", type=str, default=None,
                 help="quality score 가중치 JSON 경로 (옵션).")
  p.add_argument("--use_gpu", action='store_true')
  p.add_argument("--seed", type=int, default=11711)

  # 디코딩
  p.add_argument("--temperature", type=float, default=1.0)
  p.add_argument("--top_p", type=float, default=0.9)
  p.add_argument("--top_k", type=int, default=0)
  p.add_argument("--max_length", type=int, default=160)
  p.add_argument("--repetition_penalty", type=float, default=1.1)
  p.add_argument("--no_repeat_ngram_size", type=int, default=3)
  p.add_argument("--max_lines", type=int, default=14)

  return p.parse_args()


if __name__ == "__main__":
  args = get_args()
  seed_everything(args.seed)
  run(args)
