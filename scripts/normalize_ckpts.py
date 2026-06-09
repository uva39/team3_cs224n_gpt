"""ckpt 파일명을 일관된 규칙으로 정리한다.

규칙
    풀FT: {model_size}-fullft-{epochs}ep-lr{lr}-sonnet.pt
    LoRA: {model_size}-lora-r{rank}-{epochs}ep-lr{lr}-sonnet.pt
    DPO : {base_model}-lora-dpo-{epochs}ep-lr{lr}-b{beta}.pt
            * DPO ckpt는 SFT의 args만 들고 있어 외부 정보(파일명/sidecar)로 식별 필요.

- ckpt를 한 개씩 cpu로 lazy 로드 후 args만 추출, 즉시 해제 → 메모리 안전.
- DPO ckpt는 args에 표시가 없으므로 파일명에 'dpo'가 포함된 경우 수동 매핑 사용.
- predictions/ 안의 train log / 생성 텍스트도 같은 tag 패턴으로 옮긴다.

사용
    python scripts/normalize_ckpts.py            # dry-run, 옮길 목록만 출력
    python scripts/normalize_ckpts.py --apply    # 실제로 mv
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
from pathlib import Path

import torch


def fmt_lr(lr) -> str:
  s = f"{float(lr):.0e}"
  return s.replace("e-0", "e-").replace("e+0", "e+")


def derive_tag(args_dict: dict) -> str:
  model = args_dict.get("model_size", "gpt2")
  use_lora = bool(args_dict.get("use_lora", False))
  epochs = args_dict.get("epochs", "?")
  lr = fmt_lr(args_dict.get("lr", 0))
  if use_lora:
    rank = args_dict.get("lora_rank", "?")
    return f"{model}-lora-r{rank}-{epochs}ep-lr{lr}"
  return f"{model}-fullft-{epochs}ep-lr{lr}"


def canonical_ckpt_name(args_dict: dict, *, dpo_meta: dict | None = None) -> str:
  if dpo_meta:
    base_model = args_dict.get("model_size", "gpt2")
    epochs = dpo_meta.get("epochs", "?")
    lr = fmt_lr(dpo_meta.get("lr", 0))
    beta = dpo_meta.get("beta", "?")
    return f"{base_model}-lora-dpo-{epochs}ep-lr{lr}-b{beta}.pt"
  return derive_tag(args_dict) + "-sonnet.pt"


# DPO ckpt는 args에서 자체 HP 복원이 안 되므로 파일명/외부 메타로 수동 매핑.
# (DPO 학습 로그 predictions/dpo_train_log.json 의 config 블록과 매칭.)
DPO_KNOWN: dict[str, dict] = {
  "gpt2-large-lora-dpo.pt": {"epochs": 5, "lr": 5e-5, "beta": 0.1},
}


def load_args(ckpt_path: Path) -> dict | None:
  saved = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
  try:
    args_obj = saved.get("args")
    if args_obj is None:
      return None
    return dict(vars(args_obj)) if hasattr(args_obj, "__dict__") else dict(args_obj)
  finally:
    del saved
    gc.collect()


def plan_ckpts(root: Path) -> list[tuple[Path, Path]]:
  plan: list[tuple[Path, Path]] = []
  for p in sorted(root.glob("*.pt")):
    if p.name == "optimizer_test.npy":
      continue
    print(f"\n[inspect] {p.name}")
    args = load_args(p)
    if args is None:
      print("  (no args dict — skip)")
      continue
    dpo_meta = DPO_KNOWN.get(p.name)
    target = canonical_ckpt_name(args, dpo_meta=dpo_meta)
    print(f"  → canonical: {target}")
    if target != p.name:
      plan.append((p, p.with_name(target)))
  return plan


def derive_tag_for_predictions(p: Path) -> str | None:
  """predictions/ 안의 train_*.json 또는 generated_*.txt 와 매칭되는 tag 도출.

  명명 가이드: train_{tag}.json / generated_{tag}.txt
  """
  name = p.name
  m_train = re.match(r"^train_(.+)\.json$", name)
  if m_train:
    return m_train.group(1)
  m_gen = re.match(r"^generated_(.+)\.txt$", name)
  if m_gen:
    return m_gen.group(1)
  return None


def plan_predictions(pred_dir: Path) -> list[tuple[Path, Path]]:
  """predictions/ 안의 ad-hoc 명명 → 규칙 명명 매핑.

  현 시점에 알려진 ad-hoc 파일들만 명시적으로 매핑. (스캔 자동화는 위험.)
  """
  mapping = {
    # 구 베이스라인 (gpt2 small 풀FT 10ep lr1e-5)
    "train_log.json": "train_gpt2-fullft-10ep-lr1e-5.json",
    # 5월 30일 LoRA 학습 로그
    "train_gpt2-large_lora.json": "train_gpt2-large-lora-r16-8ep-lr2e-4.json",
    # 방금 새 small 풀FT 15ep 로그 (이미 깔끔하지만 새 규칙에 맞춤)
    "train_gpt2small_fullft.json": "train_gpt2-fullft-15ep-lr1e-5.json",
    # DPO 학습 로그
    "dpo_train_log.json": "train_gpt2-large-lora-dpo-5ep-lr5e-5-b0.1.json",
    # generated_sonnets.txt — sonnet_generation.py 기본 출력. 현재 보유분은
    # 방금 small 풀FT 15ep 마지막에 자동 생성된 것.
    "generated_sonnets.txt": "generated_gpt2-fullft-15ep-lr1e-5.txt",
    # SFT+DPO 앙상블 제출 파일 (held-out test)
    "generated_sonnets_ensemble.txt": "submit_ensemble_sft+dpo.txt",
    # dev set 평가용 ensemble 출력
    "ensemble_dev.txt": "dev_ensemble_sft+dpo.txt",
  }
  plan: list[tuple[Path, Path]] = []
  for old, new in mapping.items():
    p = pred_dir / old
    if p.exists():
      plan.append((p, pred_dir / new))
  return plan


def apply_plan(plan: list[tuple[Path, Path]], *, apply: bool):
  if not plan:
    print("  (nothing to rename)")
    return
  print()
  for src, dst in plan:
    if dst.exists():
      print(f"  SKIP (target exists): {src.name} → {dst.name}")
      continue
    print(f"  {'MOVE' if apply else 'DRY '} {src.name}  →  {dst.name}")
    if apply:
      os.rename(src, dst)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--apply", action="store_true", help="실제로 mv 실행 (기본 dry-run)")
  ap.add_argument("--root", type=str, default=".", help="프로젝트 루트")
  args = ap.parse_args()

  root = Path(args.root).resolve()
  pred = root / "predictions"

  print(f"=== normalize_ckpts (apply={args.apply}) ===")
  print(f"root: {root}")

  print("\n--- 1. ckpt (*.pt) ---")
  ckpt_plan = plan_ckpts(root)
  print("\n[rename plan: ckpt]")
  apply_plan(ckpt_plan, apply=args.apply)

  if pred.is_dir():
    print("\n--- 2. predictions/ logs & sonnets ---")
    pred_plan = plan_predictions(pred)
    print("\n[rename plan: predictions/]")
    apply_plan(pred_plan, apply=args.apply)
  else:
    print("\n(no predictions/ dir found)")

  if not args.apply:
    print("\n(dry-run only. 적용하려면 --apply 옵션 추가)")


if __name__ == "__main__":
  main()
