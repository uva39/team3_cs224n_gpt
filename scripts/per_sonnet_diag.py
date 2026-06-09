"""Per-sonnet diagnostic — 여러 ckpt에 대해 dev 12편을 한 편씩 CHRF로 평가하고
형식 통계(줄 수, 평균 줄 길이, last-word 운율 일치 등)를 같이 기록한다.

목적: corpus-level CHRF의 ±0.3 노이즈가 어디서 나오는지 sonnet-level로 분해.
어떤 prompt가 항상 어렵고 어떤 prompt가 method에 따라 갈리는지 보여 method-effect
검증의 한계(데이터 12편)를 정량화한다.

산출: predictions/per_sonnet_diag.csv  (long format: ckpt × sonnet_id × chrf 등).
"""

import argparse
import csv
import gc
import os
import re
import sys
import time
from types import SimpleNamespace

import torch
from sacrebleu.metrics import CHRF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import SonnetsDataset  # noqa: E402
from sonnet_generation import SonnetGPT, seed_everything  # noqa: E402


def _line_stats(text):
  lines = [ln.strip() for ln in text.strip().split('\n') if ln.strip()]
  n = len(lines)
  if n == 0:
    return {'n_lines': 0, 'avg_line_len': 0.0, 'n_chars': 0}
  return {
    'n_lines': n,
    'avg_line_len': sum(len(ln) for ln in lines) / n,
    'n_chars': sum(len(ln) for ln in lines),
  }


def _eval_one_ckpt(ckpt_path, prompts, golds, decode_cfg, device, seed):
  """ckpt 1개를 로드해 모든 prompt에 generate, 결과를 (id, chrf, stats) 리스트로 반환."""
  saved = torch.load(ckpt_path, map_location='cpu', weights_only=False)
  model = SonnetGPT(saved['args'], init_only=True)
  model.load_state_dict(saved['model'])
  del saved
  gc.collect()
  model = model.to(device).eval()
  torch.cuda.empty_cache()

  chrf_metric = CHRF()
  rows = []
  for (sid, prompt_text), (_, gold_text) in zip(prompts, golds):
    seed_everything(seed)  # 같은 seed로 reproducibility (per-prompt 비교 noise 제거).
    enc = model.tokenizer(prompt_text, return_tensors='pt', padding=False,
                          truncation=True).to(device)
    _, decoded = model.generate(
      enc['input_ids'],
      temperature=decode_cfg.temperature,
      top_p=decode_cfg.top_p,
      max_length=decode_cfg.max_length,
      top_k=0,
      repetition_penalty=decode_cfg.repetition_penalty,
      no_repeat_ngram_size=decode_cfg.no_repeat_ngram_size,
      max_lines=decode_cfg.max_lines,
    )
    # CHRF: 한 편 짜리 corpus_score (== sentence_score 효과).
    score = float(chrf_metric.corpus_score([decoded], [[gold_text]]).score)
    stats = _line_stats(decoded)
    rows.append({
      'sonnet_id': sid,
      'chrf': score,
      **stats,
      'gold_n_lines': len(gold_text.strip().split('\n')),
    })

  # 정리.
  model.cpu()
  del model
  gc.collect()
  torch.cuda.empty_cache()
  return rows


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--ckpts', type=str, required=True,
                 help='콤마구분 ckpt 경로 (label=path 형식도 OK).')
  p.add_argument('--prompt_path', type=str, default='data/json/sonnets_held_out_dev.json')
  p.add_argument('--gold_path', type=str, default='data/json/TRUE_sonnets_held_out_dev.json')
  p.add_argument('--out_csv', type=str, default='predictions/per_sonnet_diag.csv')
  p.add_argument('--use_gpu', action='store_true')
  p.add_argument('--seed', type=int, default=11711)
  # decoding (default = sweep_decoding에서 발견한 best)
  p.add_argument('--temperature', type=float, default=1.1)
  p.add_argument('--top_p', type=float, default=0.9)
  p.add_argument('--repetition_penalty', type=float, default=1.0)
  p.add_argument('--no_repeat_ngram_size', type=int, default=3)
  p.add_argument('--max_lines', type=int, default=14)
  p.add_argument('--max_length', type=int, default=160)
  args = p.parse_args()

  device = torch.device('cuda' if args.use_gpu else 'cpu')
  decode_cfg = SimpleNamespace(
    temperature=args.temperature, top_p=args.top_p,
    repetition_penalty=args.repetition_penalty,
    no_repeat_ngram_size=args.no_repeat_ngram_size,
    max_lines=args.max_lines, max_length=args.max_length,
  )

  prompts = list(SonnetsDataset(args.prompt_path))
  golds = list(SonnetsDataset(args.gold_path))
  assert len(prompts) == len(golds), f'{len(prompts)} vs {len(golds)}'
  print(f"[diag] {len(prompts)} prompts, decode={vars(decode_cfg)}")

  ckpt_specs = []
  for spec in args.ckpts.split(','):
    spec = spec.strip()
    if '=' in spec:
      label, path = spec.split('=', 1)
    else:
      path = spec
      label = os.path.basename(path).replace('-sonnet.pt', '')
    ckpt_specs.append((label, path))

  all_rows = []
  for label, path in ckpt_specs:
    t0 = time.time()
    print(f"[diag] ckpt={label}  ({path})")
    rows = _eval_one_ckpt(path, prompts, golds, decode_cfg, device, args.seed)
    for r in rows:
      r['ckpt'] = label
      all_rows.append(r)
    mean_chrf = sum(r['chrf'] for r in rows) / len(rows)
    n_with_14 = sum(1 for r in rows if r['n_lines'] == 14)
    print(f"  done in {time.time()-t0:.0f}s  "
          f"mean_chrf={mean_chrf:.3f}  14lines={n_with_14}/{len(rows)}")

  os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
  fieldnames = ['ckpt', 'sonnet_id', 'chrf', 'n_lines', 'avg_line_len', 'n_chars', 'gold_n_lines']
  with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in all_rows:
      w.writerow({k: r[k] for k in fieldnames})
  print(f"[diag] wrote {len(all_rows)} rows to {args.out_csv}")


if __name__ == '__main__':
  main()
