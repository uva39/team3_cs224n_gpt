from pathlib import Path
import json
import csv

RUNS_DIR = Path("./runs")

rows = []

for path in RUNS_DIR.rglob("*summary.json"):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = data.get("dataset", "unknown")

        row = {
            "dataset": dataset,
            "run_group": path.parent.name,
            "filename": path.name,
            "path": str(path),

            "best_dev_acc": data.get("best_dev_acc"),
            "best_dev_f1": data.get("best_dev_f1"),
            "best_epoch": data.get("best_epoch"),

            "final_train_acc": data.get("final_train_acc"),
            "final_train_f1": data.get("final_train_f1"),
            "final_dev_acc": data.get("final_dev_acc"),
            "final_dev_f1": data.get("final_dev_f1"),

            "fine_tune_mode": data.get("fine_tune_mode"),
            "lr": data.get("lr"),
            "epochs": data.get("epochs"),
            "batch_size": data.get("batch_size"),
            "hidden_dropout_prob": data.get("hidden_dropout_prob"),

            "pooling_config": data.get("pooling_config"),
            "use_simple_classifier": data.get("use_simple_classifier"),
            "use_rdrop": data.get("use_rdrop"),
            "rdrop_alpha": data.get("rdrop_alpha"),

            "weight_decay": data.get("weight_decay"),
            "max_grad_norm": data.get("max_grad_norm"),
            "unuse_schedule": data.get("unuse_schedule"),
            "warmup_ratio": data.get("warmup_ratio"),

            "prediction_prefix": data.get("prediction_prefix"),
            "checkpoint_path": data.get("checkpoint_path"),
        }

        rows.append(row)

    except Exception as e:
        print(f"[ERROR] {path}: {e}")


def safe_float(x, default=-1.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def save_csv(path, rows):
    fieldnames = [
        "dataset",
        "best_dev_acc",
        "best_dev_f1",
        "best_epoch",
        "final_train_acc",
        "final_train_f1",
        "final_dev_acc",
        "final_dev_f1",
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
        "run_group",
        "filename",
        "prediction_prefix",
        "checkpoint_path",
        "path",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_top(title, rows, n=15):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    for i, r in enumerate(rows[:n], start=1):
        print(
            f"{i:02d} | "
            f"acc={safe_float(r['best_dev_acc']):.6f} | "
            f"f1={safe_float(r['best_dev_f1']):.6f} | "
            f"epoch={r['best_epoch']} | "
            f"mode={r['fine_tune_mode']} | "
            f"lr={r['lr']} | "
            f"pool={r['pooling_config']} | "
            f"simple={r['use_simple_classifier']} | "
            f"rdrop={r['use_rdrop']} | "
            f"alpha={r['rdrop_alpha']} | "
            f"{r['path']}"
        )


for dataset in ["SST", "cfimdb"]:
    dataset_rows = [r for r in rows if r["dataset"] == dataset]

    by_acc = sorted(
        dataset_rows,
        key=lambda r: (
            safe_float(r["best_dev_acc"]),
            safe_float(r["best_dev_f1"]),
        ),
        reverse=True,
    )

    by_f1 = sorted(
        dataset_rows,
        key=lambda r: (
            safe_float(r["best_dev_f1"]),
            safe_float(r["best_dev_acc"]),
        ),
        reverse=True,
    )

    print_top(f"{dataset.upper()} TOP by best_dev_acc", by_acc)
    print_top(f"{dataset.upper()} TOP by best_dev_f1", by_f1)

    save_csv(f"{dataset}_runs_sorted_by_acc.csv", by_acc)
    save_csv(f"{dataset}_runs_sorted_by_f1.csv", by_f1)


print("\nSaved CSV files:")
print("- sst_runs_sorted_by_acc.csv")
print("- sst_runs_sorted_by_f1.csv")
print("- cfimdb_runs_sorted_by_acc.csv")
print("- cfimdb_runs_sorted_by_f1.csv")