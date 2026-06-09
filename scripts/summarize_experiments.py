"""Summarize all training runs by scanning predictions/train_*.json.

각 run의 핵심 지표(best dev CHRF, best epoch, history 길이) + 학습 구성을
한 표로 정리한다. 정렬 기본은 best_dev_chrf 내림차순.

사용:
    python scripts/summarize_experiments.py                       # 콘솔 표
    python scripts/summarize_experiments.py --csv out.csv         # CSV 저장
    python scripts/summarize_experiments.py --sort lr             # 컬럼 기준 정렬
    python scripts/summarize_experiments.py --filter model_size=gpt2-medium
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


COLUMNS = [
    "tag",
    "model_size",
    "method",          # fullft / lora-r{N}
    "seed",
    "lr",
    "batch_size",
    "epochs",
    "best_epoch",
    "best_dev_chrf",
    "weight_decay",
    "hidden_dropout",
    "attn_dropout",
    "grad_checkpoint",
    "ckpt_path",
    "log_path",
    "out_path",
]


def parse_log(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[warn] cannot read {path}: {e}", file=sys.stderr)
        return None
    if not isinstance(payload, dict):
        return None
    cfg = payload.get("config", {}) or {}
    base = os.path.basename(path)
    if base.startswith("train_"):
        base = base[len("train_"):]
    if base.endswith(".json"):
        base = base[: -len(".json")]
    tag = base
    method = (
        f"lora-r{cfg.get('lora_rank')}" if cfg.get("use_lora") else "fullft"
    )
    # ckpt 경로 추정(새 규칙). 옛 run은 repo root의 다른 이름일 수 있다.
    ckpt_path = f"predictions/ckpts/{tag}-sonnet.pt"
    if not os.path.exists(ckpt_path):
        legacy = f"{tag}-sonnet.pt"
        if os.path.exists(legacy):
            ckpt_path = legacy
        else:
            ckpt_path += " (missing)"
    out_path = f"predictions/generated_{tag}.txt"
    if not os.path.exists(out_path):
        out_path += " (missing)"
    return {
        "tag": tag,
        "model_size": cfg.get("model_size"),
        "method": method,
        "seed": cfg.get("seed"),
        "lr": cfg.get("lr"),
        "batch_size": cfg.get("batch_size"),
        "epochs": cfg.get("epochs"),
        "best_epoch": payload.get("best_epoch"),
        "best_dev_chrf": payload.get("best_dev_chrf"),
        "weight_decay": cfg.get("weight_decay"),
        "hidden_dropout": cfg.get("hidden_dropout"),
        "attn_dropout": cfg.get("attn_dropout"),
        "grad_checkpoint": cfg.get("grad_checkpoint"),
        "ckpt_path": ckpt_path,
        "log_path": path,
        "out_path": out_path,
    }


def format_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "(no rows)"
    # numerics formatted compactly
    display_rows = []
    for r in rows:
        d = {}
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                # lr 같은 매우 작은 값은 지수표기, CHRF처럼 보통 값은 소수점.
                if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
                    d[c] = f"{v:.2e}"
                else:
                    d[c] = f"{v:.4f}"
            elif v is None:
                d[c] = "-"
            else:
                d[c] = str(v)
        display_rows.append(d)
    widths = {c: max(len(c), max(len(d[c]) for d in display_rows)) for c in cols}
    sep = "  "
    head = sep.join(c.ljust(widths[c]) for c in cols)
    bar = sep.join("-" * widths[c] for c in cols)
    body = "\n".join(sep.join(d[c].ljust(widths[c]) for c in cols) for d in display_rows)
    return f"{head}\n{bar}\n{body}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="predictions/train_*.json",
                    help="대상 로그 파일 glob (기본 predictions/train_*.json).")
    ap.add_argument("--sort", default="best_dev_chrf",
                    help="정렬 기준 컬럼명 (기본 best_dev_chrf).")
    ap.add_argument("--asc", action="store_true", help="오름차순 정렬.")
    ap.add_argument("--filter", action="append", default=[],
                    help="key=value 필터. 여러 번 지정 가능.")
    ap.add_argument("--csv", default=None, help="CSV 출력 경로.")
    ap.add_argument("--cols", default=None,
                    help="콤마구분 컬럼 목록 (기본=전체). 'short' 키워드는 간단 표.")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"[empty] no logs matched {args.glob}", file=sys.stderr)
        return 0

    rows = [r for r in (parse_log(p) for p in files) if r is not None]

    # filter
    for f in args.filter:
        if "=" not in f:
            print(f"[warn] ignoring bad filter: {f}", file=sys.stderr)
            continue
        k, v = f.split("=", 1)
        rows = [r for r in rows if str(r.get(k)) == v]

    # sort
    def keyfn(r):
        v = r.get(args.sort)
        # None은 항상 끝으로
        if v is None:
            return (1, 0)
        if isinstance(v, (int, float)):
            return (0, -float(v) if not args.asc else float(v))
        return (0, str(v))
    rows.sort(key=keyfn)
    if not args.asc:
        # 위의 키가 음수로 내림차순을 이미 표현했으므로 다시 뒤집지 않는다.
        pass

    cols = COLUMNS if args.cols is None else (
        ["tag", "model_size", "method", "seed", "lr", "best_epoch", "best_dev_chrf"]
        if args.cols == "short" else [c.strip() for c in args.cols.split(",") if c.strip()]
    )

    print(format_table(rows, cols))
    print(f"\n[{len(rows)} runs]")

    if args.csv:
        import csv
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c in COLUMNS})
        print(f"[csv] wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
