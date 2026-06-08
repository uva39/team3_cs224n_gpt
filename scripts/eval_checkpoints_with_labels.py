#!/usr/bin/env python3

"""
Evaluate saved sentiment-classification checkpoints on labeled test-student files.

This script:
1. Scans runs/**/*summary.json
2. Reads each summary's checkpoint_path and experiment metadata
3. Loads the corresponding checkpoint
4. Reconstructs GPT2SentimentClassifier from saved model_config
5. Loads ids-sst-test-student.csv / ids-cfimdb-test-student.csv as labeled data
6. Computes test accuracy and macro-F1
7. Saves dataset-specific sorted CSV files

Recommended usage:

  PYTHONPATH=. python -u scripts/eval_checkpoints_with_labels.py \
      --use_gpu \
      --summary-glob "runs/**/*summary.json" \
      --skip-ambiguous-checkpoints \
      --save-preds

If older experiments reused the same checkpoint path such as checkpoints/sst.pt,
use --skip-ambiguous-checkpoints to avoid evaluating overwritten checkpoints.
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from classifier_upgrade import (
    GPT2SentimentClassifier,
    SentimentDataset,
    load_data,
)


def torch_load_checkpoint(path: Path, device):
    """
    torch.load compatibility wrapper.
    Some PyTorch versions support weights_only; older ones do not.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_dataset_name(name, fallback_path=None):
    """
    Normalize dataset names from summary JSON.
    Some old summaries may use "SST" while others use "sst".
    """
    if name is not None:
        lowered = str(name).lower()
        if lowered == "sst":
            return "sst"
        if lowered == "cfimdb":
            return "cfimdb"

    if fallback_path is not None:
        lowered_path = str(fallback_path).lower()
        if "cfimdb" in lowered_path:
            return "cfimdb"
        if "sst" in lowered_path:
            return "sst"

    raise ValueError(f"Unknown dataset name: {name}")


def as_namespace(obj):
    if obj is None:
        return SimpleNamespace()
    if isinstance(obj, SimpleNamespace):
        return obj
    if isinstance(obj, dict):
        return SimpleNamespace(**obj)
    return obj


def count_num_labels(labeled_data):
    labels = [row[1] for row in labeled_data]
    return len(set(labels))


def fill_config_from_summary(saved_config, summary, dataset_slug, num_labels):
    """
    Reconstruct model config robustly.

    Older checkpoints may not contain newer fields such as:
    - pooling_config
    - use_simple_classifier

    This function fills missing values from summary JSON or safe defaults.
    """
    config = as_namespace(saved_config)

    defaults = {
        "hidden_dropout_prob": summary.get("hidden_dropout_prob", 0.2),
        "num_labels": num_labels,
        "hidden_size": 768,
        "data_dir": ".",
        "fine_tune_mode": summary.get("fine_tune_mode", "full-model"),
        "use_simple_classifier": summary.get("use_simple_classifier", True),
        "pooling_config": summary.get("pooling_config", "last"),
    }

    for key, value in defaults.items():
        if not hasattr(config, key) or getattr(config, key) is None:
            setattr(config, key, value)

    # Test set label count should be trusted for output dimension.
    # Usually this matches the saved config.
    config.num_labels = num_labels

    return config


@torch.no_grad()
def eval_labeled_dataset(model, dataloader, device):
    model.eval()

    y_true = []
    y_pred = []
    sent_ids = []
    sents = []

    for batch in tqdm(dataloader, desc="test-eval", leave=False):
        b_ids = batch["token_ids"].to(device)
        b_mask = batch["attention_mask"].to(device)
        b_labels = batch["labels"].detach().cpu().numpy().tolist()

        logits = model(b_ids, b_mask)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()

        y_true.extend(b_labels)
        y_pred.extend(preds)
        sent_ids.extend(batch["sent_ids"])
        sents.extend(batch["sents"])

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return acc, f1, y_true, y_pred, sent_ids, sents


def read_summary(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "test_acc",
        "test_f1",
        "num_test_examples",

        "best_dev_acc",
        "best_dev_f1",
        "best_epoch",
        "final_dev_acc",
        "final_dev_f1",
        "final_train_acc",
        "final_train_f1",

        "seed",
        "fine_tune_mode",
        "lr",
        "epochs",
        "batch_size",
        "hidden_dropout_prob",

        "pooling_config",
        "use_simple_classifier",
        "use_rdrop",
        "rdrop_alpha",
        "weight_decay",
        "max_grad_norm",
        "unuse_schedule",
        "warmup_ratio",

        "prediction_prefix",
        "checkpoint_path",
        "summary_path",
        "predictions_out",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path: Path, sent_ids, sents, y_true, y_pred):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "sentence",
                "true_label",
                "pred_label",
                "correct",
            ],
        )
        writer.writeheader()

        for sid, sent, true, pred in zip(sent_ids, sents, y_true, y_pred):
            writer.writerow({
                "id": sid,
                "sentence": sent,
                "true_label": true,
                "pred_label": pred,
                "correct": int(true == pred),
            })


def build_result_row(
    summary,
    summary_path,
    checkpoint_path,
    dataset_slug,
    test_acc=None,
    test_f1=None,
    num_test_examples=None,
    predictions_out=None,
    error=None,
):
    return {
        "dataset": dataset_slug,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "num_test_examples": num_test_examples,

        "best_dev_acc": summary.get("best_dev_acc"),
        "best_dev_f1": summary.get("best_dev_f1"),
        "best_epoch": summary.get("best_epoch"),
        "final_dev_acc": summary.get("final_dev_acc"),
        "final_dev_f1": summary.get("final_dev_f1"),
        "final_train_acc": summary.get("final_train_acc"),
        "final_train_f1": summary.get("final_train_f1"),

        "seed": summary.get("seed"),
        "fine_tune_mode": summary.get("fine_tune_mode"),
        "lr": summary.get("lr"),
        "epochs": summary.get("epochs"),
        "batch_size": summary.get("batch_size"),
        "hidden_dropout_prob": summary.get("hidden_dropout_prob"),

        "pooling_config": summary.get("pooling_config"),
        "use_simple_classifier": summary.get("use_simple_classifier"),
        "use_rdrop": summary.get("use_rdrop"),
        "rdrop_alpha": summary.get("rdrop_alpha"),
        "weight_decay": summary.get("weight_decay"),
        "max_grad_norm": summary.get("max_grad_norm"),
        "unuse_schedule": summary.get("unuse_schedule"),
        "warmup_ratio": summary.get("warmup_ratio"),

        "prediction_prefix": summary.get("prediction_prefix"),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "summary_path": str(summary_path),
        "predictions_out": str(predictions_out) if predictions_out is not None else None,
        "error": error,
    }


def safe_float(value, default=-1.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def collect_summaries(summary_glob):
    summary_paths = sorted(ROOT_DIR.glob(summary_glob))
    summaries = []

    for summary_path in summary_paths:
        try:
            summary = read_summary(summary_path)

            checkpoint_path = summary.get("checkpoint_path")
            if checkpoint_path is None:
                print(f"[SKIP] no checkpoint_path: {summary_path}")
                continue

            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.is_absolute():
                checkpoint_path = ROOT_DIR / checkpoint_path

            dataset_slug = normalize_dataset_name(
                summary.get("dataset"),
                fallback_path=summary_path,
            )

            summaries.append({
                "summary": summary,
                "summary_path": summary_path,
                "checkpoint_path": checkpoint_path,
                "dataset": dataset_slug,
            })

        except Exception as e:
            print(f"[ERROR] failed to parse summary {summary_path}: {e}")

    return summaries


def print_ambiguous_checkpoints(summaries):
    ckpt_to_summaries = defaultdict(list)

    for item in summaries:
        ckpt_to_summaries[str(item["checkpoint_path"])].append(str(item["summary_path"]))

    ambiguous = {
        ckpt: paths
        for ckpt, paths in ckpt_to_summaries.items()
        if len(paths) > 1
    }

    if not ambiguous:
        return ckpt_to_summaries, ambiguous

    print("\n[WARNING] Multiple summaries share the same checkpoint path.")
    print("These checkpoints may have been overwritten by later runs.")
    print("Use --skip-ambiguous-checkpoints to skip them.")
    print(f"Ambiguous checkpoint count: {len(ambiguous)}")

    for ckpt, paths in list(ambiguous.items())[:10]:
        print(f"\n- {ckpt}")
        for path in paths[:5]:
            print(f"    {path}")
        if len(paths) > 5:
            print(f"    ... and {len(paths) - 5} more")

    print()

    return ckpt_to_summaries, ambiguous


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary-glob",
        default="runs/**/*summary.json",
        help='Glob pattern under project root. Example: "runs/**/*summary.json"',
    )
    parser.add_argument(
        "--output-dir",
        default="reports/test_eval",
        help="Directory to save evaluation CSVs.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override eval batch size. If omitted, summary batch_size is used.",
    )
    parser.add_argument(
        "--skip-ambiguous-checkpoints",
        action="store_true",
        help="Skip checkpoints shared by multiple summaries.",
    )
    parser.add_argument(
        "--save-preds",
        action="store_true",
        help="Save per-example test predictions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N summaries for debugging.",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "sst", "cfimdb"],
        default="all",
        help="Evaluate only one dataset or all.",
    )

    args = parser.parse_args()

    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")

    print("============================================================")
    print("Evaluate checkpoints with labeled test set")
    print("============================================================")
    print(f"ROOT_DIR     = {ROOT_DIR}")
    print(f"device       = {device}")
    print(f"summary_glob = {args.summary_glob}")
    print(f"output_dir   = {output_dir}")
    print("============================================================")

    summaries = collect_summaries(args.summary_glob)

    if args.dataset != "all":
        summaries = [item for item in summaries if item["dataset"] == args.dataset]

    if args.limit is not None:
        summaries = summaries[:args.limit]

    print(f"Found {len(summaries)} summary files to evaluate.")

    ckpt_to_summaries, ambiguous = print_ambiguous_checkpoints(summaries)

    rows = []
    error_rows = []

    for idx, item in enumerate(summaries, start=1):
        summary = item["summary"]
        summary_path = item["summary_path"]
        checkpoint_path = item["checkpoint_path"]
        dataset_slug = item["dataset"]

        if args.skip_ambiguous_checkpoints and str(checkpoint_path) in ambiguous:
            print(f"[SKIP ambiguous] {checkpoint_path}")
            continue

        if not checkpoint_path.exists():
            msg = "missing checkpoint"
            print(f"[SKIP missing checkpoint] {checkpoint_path}")
            error_rows.append(build_result_row(
                summary=summary,
                summary_path=summary_path,
                checkpoint_path=checkpoint_path,
                dataset_slug=dataset_slug,
                error=msg,
            ))
            continue

        test_path = ROOT_DIR / f"data/ids-{dataset_slug}-test-student.csv"

        if not test_path.exists():
            msg = f"missing test file: {test_path}"
            print(f"[SKIP missing test file] {test_path}")
            error_rows.append(build_result_row(
                summary=summary,
                summary_path=summary_path,
                checkpoint_path=checkpoint_path,
                dataset_slug=dataset_slug,
                error=msg,
            ))
            continue

        try:
            # The user confirmed that test-student files have labels.
            # Therefore, load them as valid/labeled data.
            test_data = load_data(str(test_path), "valid")
            num_labels = count_num_labels(test_data)

            checkpoint = torch_load_checkpoint(checkpoint_path, device)

            saved_config = checkpoint.get("model_config", None)
            config = fill_config_from_summary(
                saved_config=saved_config,
                summary=summary,
                dataset_slug=dataset_slug,
                num_labels=num_labels,
            )

            eval_batch_size = args.batch_size
            if eval_batch_size is None:
                eval_batch_size = summary.get("batch_size", 8)
                if eval_batch_size is None:
                    eval_batch_size = 8

            test_dataset = SentimentDataset(test_data, config)
            test_dataloader = DataLoader(
                test_dataset,
                shuffle=False,
                batch_size=int(eval_batch_size),
                collate_fn=test_dataset.collate_fn,
            )

            print("\n============================================================")
            print(f"[{idx}/{len(summaries)}] Evaluating")
            print(f"dataset    : {dataset_slug}")
            print(f"summary    : {summary_path}")
            print(f"checkpoint : {checkpoint_path}")
            print(f"mode       : {summary.get('fine_tune_mode')}")
            print(f"lr         : {summary.get('lr')}")
            print(f"seed       : {summary.get('seed')}")
            print(f"pooling    : {getattr(config, 'pooling_config', None)}")
            print(f"simple     : {getattr(config, 'use_simple_classifier', None)}")
            print(f"rdrop      : {summary.get('use_rdrop')}")
            print(f"alpha      : {summary.get('rdrop_alpha')}")
            print("============================================================")

            model = GPT2SentimentClassifier(config)
            model.load_state_dict(checkpoint["model"])
            model = model.to(device)

            test_acc, test_f1, y_true, y_pred, sent_ids, sents = eval_labeled_dataset(
                model=model,
                dataloader=test_dataloader,
                device=device,
            )

            print(f"test_acc={test_acc:.6f}, test_f1={test_f1:.6f}")

            predictions_out = None
            if args.save_preds:
                pred_name = summary_path.stem.replace("/", "_") + "-labeled-test-preds.csv"
                predictions_out = output_dir / "predictions" / pred_name
                write_predictions(
                    path=predictions_out,
                    sent_ids=sent_ids,
                    sents=sents,
                    y_true=y_true,
                    y_pred=y_pred,
                )
                print(f"saved predictions: {predictions_out}")

            rows.append(build_result_row(
                summary=summary,
                summary_path=summary_path,
                checkpoint_path=checkpoint_path,
                dataset_slug=dataset_slug,
                test_acc=float(test_acc),
                test_f1=float(test_f1),
                num_test_examples=len(test_data),
                predictions_out=predictions_out,
            ))

        except Exception as e:
            msg = repr(e)
            print(f"[ERROR] {summary_path}: {msg}")
            error_rows.append(build_result_row(
                summary=summary,
                summary_path=summary_path,
                checkpoint_path=checkpoint_path,
                dataset_slug=dataset_slug,
                error=msg,
            ))

    # Save all results.
    all_rows_by_dataset_acc = sorted(
        rows,
        key=lambda r: (
            r["dataset"],
            -safe_float(r["test_acc"]),
            -safe_float(r["test_f1"]),
        ),
    )
    write_csv(output_dir / "test_checkpoint_eval_results.csv", all_rows_by_dataset_acc)

    # Save error rows too.
    if error_rows:
        write_csv(output_dir / "test_checkpoint_eval_errors.csv", error_rows)

    # Save dataset-specific sorted results.
    for dataset_slug in ["sst", "cfimdb"]:
        ds_rows = [r for r in rows if r["dataset"] == dataset_slug]

        by_acc = sorted(
            ds_rows,
            key=lambda r: (
                safe_float(r["test_acc"]),
                safe_float(r["test_f1"]),
            ),
            reverse=True,
        )

        by_f1 = sorted(
            ds_rows,
            key=lambda r: (
                safe_float(r["test_f1"]),
                safe_float(r["test_acc"]),
            ),
            reverse=True,
        )

        write_csv(output_dir / f"test_{dataset_slug}_sorted_by_acc.csv", by_acc)
        write_csv(output_dir / f"test_{dataset_slug}_sorted_by_f1.csv", by_f1)

        print("\n" + "=" * 100)
        print(f"{dataset_slug.upper()} TOP 10 by test_acc")
        print("=" * 100)

        for rank, r in enumerate(by_acc[:10], start=1):
            print(
                f"{rank:02d} | "
                f"test_acc={safe_float(r['test_acc']):.6f} | "
                f"test_f1={safe_float(r['test_f1']):.6f} | "
                f"dev_acc={r['best_dev_acc']} | "
                f"dev_f1={r['best_dev_f1']} | "
                f"seed={r['seed']} | "
                f"lr={r['lr']} | "
                f"pool={r['pooling_config']} | "
                f"simple={r['use_simple_classifier']} | "
                f"rdrop={r['use_rdrop']} | "
                f"alpha={r['rdrop_alpha']} | "
                f"{r['summary_path']}"
            )

    print("\n============================================================")
    print("Saved files")
    print("============================================================")
    print(f"- {output_dir / 'test_checkpoint_eval_results.csv'}")
    print(f"- {output_dir / 'test_sst_sorted_by_acc.csv'}")
    print(f"- {output_dir / 'test_sst_sorted_by_f1.csv'}")
    print(f"- {output_dir / 'test_cfimdb_sorted_by_acc.csv'}")
    print(f"- {output_dir / 'test_cfimdb_sorted_by_f1.csv'}")
    if error_rows:
        print(f"- {output_dir / 'test_checkpoint_eval_errors.csv'}")
    print("============================================================")


if __name__ == "__main__":
    main()