"""
디코딩 하이퍼파라미터 스윕 (재학습 불필요).

체크포인트 1개를 로드한 뒤 (temperature, top_p, top_k, repetition_penalty,
no_repeat_ngram_size, max_lines) 그리드 위에서 dev CHRF를 측정한다.
재학습이 필요 없으므로 매우 저렴하고, CHRF가 가장 크게 흔들리는 축이
보통 temperature/top_p 라 효과가 크다.

산출물: predictions/decoding_sweep.csv (best 순 정렬).

예시:
  python scripts/sweep_decoding.py --use_gpu \
         --ckpt gpt2-large-lora-8ep-0.0002-sonnet.pt \
         --temperatures 0.7,0.9,1.0,1.1,1.2 \
         --top_ps 0.85,0.9,0.95 \
         --rep_penalties 1.0,1.1,1.2 \
         --no_repeat_ngrams 0,3 \
         --max_lines 14
"""

import argparse
import csv
import gc
import itertools
import os
import sys
import time
from types import SimpleNamespace

import torch

# scripts/ 에서 실행해도 import가 되도록 부모 경로 추가.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sonnet_generation import SonnetGPT, evaluate_dev_chrf, seed_everything  # noqa: E402


def parse_list(s, cast):
  return [cast(x.strip()) for x in s.split(',') if x.strip()]


def run(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.ckpt, map_location='cpu', weights_only=False)
  # init_only=True: HF gpt2-large 가중치 로드 스킵 → cgroup 9GB 환경에서 OOM 회피.
  model = SonnetGPT(saved['args'], init_only=True)
  model.load_state_dict(saved['model'])
  del saved
  gc.collect()
  model = model.to(device).eval()
  if args.use_gpu:
    torch.cuda.empty_cache()

  temps = parse_list(args.temperatures, float)
  tops = parse_list(args.top_ps, float)
  topks = parse_list(args.top_ks, int)
  reps = parse_list(args.rep_penalties, float)
  nrns = parse_list(args.no_repeat_ngrams, int)

  grid = list(itertools.product(temps, tops, topks, reps, nrns))
  print(f"[sweep] {len(grid)} configs, ckpt={args.ckpt}")

  results = []
  t0 = time.time()
  for i, (T, P, K, R, N) in enumerate(grid):
    cfg = SimpleNamespace(
      temperature=T, top_p=P, top_k=K, repetition_penalty=R,
      no_repeat_ngram_size=N, max_lines=args.max_lines,
    )
    chrf, _ = evaluate_dev_chrf(
      model, cfg, device,
      prompt_path=args.dev_prompt_path,
      gold_path=args.dev_gold_path,
      max_length=args.dev_max_new_tokens,
      verbose=False,
    )
    elapsed = time.time() - t0
    print(f"[{i+1}/{len(grid)}] T={T} top_p={P} top_k={K} rep={R} nrn={N} max_lines={args.max_lines} "
          f"-> chrf={chrf:.4f}  (elapsed {elapsed:.0f}s)")
    results.append({
      'temperature': T, 'top_p': P, 'top_k': K,
      'repetition_penalty': R, 'no_repeat_ngram_size': N,
      'max_lines': args.max_lines, 'dev_chrf': chrf,
    })
    gc.collect()
    if args.use_gpu:
      torch.cuda.empty_cache()

  results.sort(key=lambda r: r['dev_chrf'], reverse=True)
  os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
  with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    for row in results:
      writer.writerow(row)
  print(f"[sweep] wrote {args.out_csv}. Best: {results[0]}")


def get_args():
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", type=str, required=True)
  p.add_argument("--out_csv", type=str, default="predictions/decoding_sweep.csv")
  p.add_argument("--dev_prompt_path", type=str, default="data/json/sonnets_held_out_dev.json")
  p.add_argument("--dev_gold_path", type=str, default="data/json/TRUE_sonnets_held_out_dev.json")
  p.add_argument("--dev_max_new_tokens", type=int, default=160)
  p.add_argument("--use_gpu", action='store_true')
  p.add_argument("--seed", type=int, default=11711)

  p.add_argument("--temperatures", type=str, default="0.9,1.0,1.1,1.2")
  p.add_argument("--top_ps", type=str, default="0.85,0.9,0.95")
  p.add_argument("--top_ks", type=str, default="0")
  p.add_argument("--rep_penalties", type=str, default="1.0,1.1,1.2")
  p.add_argument("--no_repeat_ngrams", type=str, default="0,3")
  p.add_argument("--max_lines", type=int, default=14)

  return p.parse_args()


if __name__ == "__main__":
  args = get_args()
  seed_everything(args.seed)
  run(args)
