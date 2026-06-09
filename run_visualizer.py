"""
run_visualizer.py

Matplotlib utilities for GPT-2 classifier experiment visualization.
Put this file next to your notebook and import functions from it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

PathLike = Union[str, Path]

DEFAULT_CATEGORICAL_DEFAULTS = {
    "fine_tune_mode": "not use",
    "pooling_config": "last",  # empty pooling_config means last pooling
    "classifier_head": "Simple head",
    "use_rdrop": "not use",
    "rdrop_alpha": "not use",
    "weight_decay": "not use",
    "max_grad_norm": "not use",
    "unuse_schedule": "not use",
    "warmup_ratio": "not use",
    "run_group": "not use",
    "lr": "not use",
    "hidden_dropout_prob": "not use",
}

DEFAULT_NUMERIC_COLS = [
    "best_dev_acc",
    "best_dev_f1",
    "best_epoch",
    "final_train_acc",
    "final_train_f1",
    "final_dev_acc",
    "final_dev_f1",
    "lr",
    "epochs",
    "batch_size",
    "hidden_dropout_prob",
    "rdrop_alpha",
    "weight_decay",
    "max_grad_norm",
    "warmup_ratio",
]


def infer_classifier_head(row):
    value = str(row.get("use_simple_classifier", "")).strip().lower()
    filename = str(row.get("filename", "")).lower()
    path = str(row.get("path", "")).lower()

    if value == "true":
        return "simple head"
    if value == "false":
        return "MLP head"

    # 예전 실험 JSON에 use_simple_classifier가 없거나 비어 있는 경우 보정
    if "simple" in filename or "simple" in path or "base" in filename or "base" in path:
        return "simple head"
    if "mlp" in filename or "mlp" in path:
        return "MLP head"

    return "unknown"


def clean_category(series: pd.Series, default: str = "not use") -> pd.Series:
    """Convert empty strings and NaN-like values into a default category."""
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "none": pd.NA})
        .fillna(default)
    )


def _as_bool_use(series: pd.Series) -> np.ndarray:
    """True -> use, False/empty -> no use."""
    s = clean_category(series).str.lower()
    return np.where(s == "true", "use", "no use")


def _numeric_option_to_use(series: pd.Series) -> np.ndarray:
    """Positive numeric value -> use, otherwise -> no use."""
    numeric = pd.to_numeric(series, errors="coerce")
    return np.where(numeric.fillna(0) > 0, "use", "no use")


def pretty_fine_tune_mode(x) -> str:
    """Human-friendly name for fine-tune mode."""
    x = str(x).strip()
    if x == "last-linear-layer":
        return "last layer only"
    if x == "full-model":
        return "full fine-tuning"
    if x == "" or x.lower() in ["nan", "none", "not use"]:
        return "not use"
    return x


def infer_experiment_group(row: pd.Series) -> str:
    """Infer a more readable experiment group from run_group/path/filename."""
    run_group = str(row.get("run_group", "")).strip()
    filename = str(row.get("filename", "")).lower()
    path = str(row.get("path", "")).lower()
    dataset = str(row.get("dataset", "")).lower()

    if run_group.lower() == "base_model":
        return "base model"

    # Some sweep runs have run_group like SST/CFIMDB, which is not informative.
    if "hyper-sweep" in path or run_group.lower() == dataset:
        if "lr-sweep" in filename:
            return "hyper sweep: lr"
        if "dropout-sweep" in filename:
            return "hyper sweep: dropout"
        if "alpha-sweep" in filename:
            return "hyper sweep: RDrop alpha"
        return "hyperparameter sweep"

    if run_group == "" or run_group.lower() in ["nan", "none"]:
        return "not use"
    return run_group


def load_runs(csv_files: Iterable[PathLike]) -> pd.DataFrame:
    """Load multiple CSV files and concatenate them."""
    csv_files = [Path(p) for p in csv_files]
    existing_files = [p for p in csv_files if p.exists()]
    if not existing_files:
        checked = ", ".join(str(p) for p in csv_files)
        raise FileNotFoundError(f"No CSV files were found. Checked: {checked}")

    dfs = []
    for path in existing_files:
        temp = pd.read_csv(path)
        temp["source_file"] = path.name
        temp["sort_metric_from_file"] = "acc" if "_acc" in path.name.lower() else "f1"
        dfs.append(temp)
    return pd.concat(dfs, ignore_index=True)


def prepare_run_dataframe(
    raw_df: pd.DataFrame,
    best_criterion: str = "best_dev_f1",
    numeric_cols: Sequence[str] = DEFAULT_NUMERIC_COLS,
    categorical_defaults: dict = DEFAULT_CATEGORICAL_DEFAULTS,
) -> pd.DataFrame:
    """Clean and enrich the raw experiment dataframe."""
    df = raw_df.copy()

    # Drop duplicates introduced by acc-sorted and f1-sorted CSVs.
    if "path" in df.columns and "dataset" in df.columns:
        df = df.drop_duplicates(subset=["dataset", "path"]).copy()
    elif "filename" in df.columns and "dataset" in df.columns:
        df = df.drop_duplicates(subset=["dataset", "filename"]).copy()
    else:
        df = df.drop_duplicates().copy()

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, default_value in categorical_defaults.items():
        if col in df.columns:
            df[col + "_cat"] = clean_category(df[col], default=default_value)

    if "fine_tune_mode" in df.columns:
        df["fine_tune_type"] = df["fine_tune_mode"].apply(pretty_fine_tune_mode)
    else:
        df["fine_tune_type"] = "not use"

    df["rdrop_use"] = _as_bool_use(df["use_rdrop"]) if "use_rdrop" in df.columns else "no use"
#    df["simple_classifier_use"] = _as_bool_use(df["use_simple_classifier"]) if "use_simple_classifier" in df.columns else "no use"
    df["classifier_head"] = df.apply(infer_classifier_head, axis=1)
    df["warmup_use"] = _numeric_option_to_use(df["warmup_ratio"]) if "warmup_ratio" in df.columns else "no use"
    df["weight_decay_use"] = _numeric_option_to_use(df["weight_decay"]) if "weight_decay" in df.columns else "no use"
    df["grad_clip_use"] = _numeric_option_to_use(df["max_grad_norm"]) if "max_grad_norm" in df.columns else "no use"

    if "unuse_schedule" in df.columns:
        s = clean_category(df["unuse_schedule"], default="false").str.lower()
        df["scheduler_use"] = np.select(
            [s == "true", s == "false"],
            ["no use", "use"],
            default="use",
        )
    else:
        # run_classifier.py 기준 기본값은 unuse_schedule=False 이므로 scheduler 사용
        df["scheduler_use"] = "no use"

    df["experiment_group"] = df.apply(infer_experiment_group, axis=1)

    if "path" in df.columns:
        df["is_hyper_sweep"] = np.where(
            df["path"].astype(str).str.contains("hyper-sweep", case=False, na=False),
            "hyperparameter sweep",
            "manual experiment",
        )
    else:
        df["is_hyper_sweep"] = "manual experiment"

    run_group_clean = clean_category(df.get("run_group", pd.Series([""] * len(df)))).str.lower()
    filename_clean = clean_category(df.get("filename", pd.Series([""] * len(df)))).str.lower()
    df["is_base_model"] = run_group_clean.eq("base_model") | filename_clean.str.contains("base", na=False)

    df["model_role"] = "others"
    df.loc[df["is_base_model"], "model_role"] = "base model"

    if best_criterion in df.columns and "dataset" in df.columns:
        valid = df.dropna(subset=[best_criterion])
        if len(valid) > 0:
            best_indices = valid.groupby("dataset")[best_criterion].idxmax()
            df.loc[best_indices, "model_role"] = "best model"
            df.loc[df["is_base_model"] & df.index.isin(best_indices), "model_role"] = "base & best"

    if "best_dev_acc" in df.columns and "best_dev_f1" in df.columns:
        df["acc_f1_gap"] = df["best_dev_acc"] - df["best_dev_f1"]

    return df


def load_and_prepare_runs(csv_files: Iterable[PathLike], best_criterion: str = "best_dev_f1") -> pd.DataFrame:
    """Convenience wrapper for load_runs() + prepare_run_dataframe()."""
    return prepare_run_dataframe(load_runs(csv_files), best_criterion=best_criterion)


# =========================
# Plot helpers
# =========================


def has_multiple_values(data: pd.DataFrame, col: str) -> bool:
    """Return True if a column has at least two non-empty values."""
    if col not in data.columns:
        return False
    values = (
        data[col]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "none": pd.NA})
        .dropna()
        .unique()
    )
    return len(values) >= 2


def _get_label_series(sub: pd.DataFrame) -> pd.Series:
    if "experiment_group" in sub.columns:
        base = sub["experiment_group"].astype(str)
    elif "run_group" in sub.columns:
        base = sub["run_group"].astype(str)
    else:
        base = pd.Series(["run"] * len(sub), index=sub.index)

    if "fine_tune_type" in sub.columns:
        base = sub["fine_tune_type"].astype(str) + " | " + base
    if "filename" in sub.columns:
        base = base + "\n" + sub["filename"].astype(str).str.slice(0, 45)
    return base


def _safe_xlim_from_values(values, x_min="auto", x_margin: float = 0.005):
    values = pd.Series(values).dropna()
    if len(values) == 0:
        return 0, 1

    metric_min = values.min()
    metric_max = values.max()

    if x_min == "auto":
        left = max(0, metric_min - x_margin)
    elif x_min is None:
        left = 0
    else:
        left = float(x_min)

    right = min(1.0, metric_max + x_margin)
    if right <= left:
        right = left + 0.01
    return left, right


def save_or_show(fig, save_path: Optional[PathLike] = None, dpi: int = 200):
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print("saved:", save_path)
    plt.show()


# =========================
# Scatter plots
# =========================


def plot_acc_f1_scatter(
    data: pd.DataFrame,
    color_by: str,
    dataset_name: Optional[str] = None,
    x_col: str = "best_dev_acc",
    y_col: str = "best_dev_f1",
    title_suffix: Optional[str] = None,
    annotate_special: bool = True,
    jitter: float = 0.0,
    special_jitter: float = 0.001,
    random_seed: int = 42,
    figsize=(9.5, 6),
    save_path: Optional[PathLike] = None,
):
    """Draw one acc-F1 scatter plot with a separate legend panel."""
    if dataset_name is not None and "dataset" in data.columns:
        sub = data[data["dataset"].astype(str) == str(dataset_name)].copy()
    else:
        sub = data.copy()

    if color_by not in sub.columns:
        print(f"skip scatter: missing column {color_by}")
        return

    sub = sub.dropna(subset=[x_col, y_col])
    if len(sub) == 0:
        print(f"skip scatter: no valid metric rows for {dataset_name or 'all data'}")
        return

    rng = np.random.default_rng(random_seed)

    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [4.5, 1.5]},
    )
    legend_ax.axis("off")

    groups = sorted(sub[color_by].astype(str).unique())
    for group in groups:
        g = sub[sub[color_by].astype(str) == group]
        x = g[x_col].to_numpy()
        y = g[y_col].to_numpy()

        if jitter > 0:
            x = x + rng.normal(0, jitter, size=len(x))
            y = y + rng.normal(0, jitter, size=len(y))

        ax.scatter(
            x,
            y,
            s=70,
            alpha=0.75,
            label=str(group),
            edgecolors="black",
            linewidths=0.4,
        )

    special_handles = []
    if annotate_special and "model_role" in sub.columns:
        special = sub[sub["model_role"].isin(["base model", "best model", "base & best"])].copy()

        for _, row in special.iterrows():
            role = row["model_role"]
            if role == "base model":
                marker, size = "X", 170
            elif role == "best model":
                marker, size = "*", 240
            else:
                marker, size = "P", 220

            x = row[x_col]
            y = row[y_col]
            if special_jitter > 0:
                x = x + rng.normal(0, special_jitter)
                y = y + rng.normal(0, special_jitter)

            ax.scatter(
                x,
                y,
                s=size,
                marker=marker,
                facecolors="none",
                edgecolors="black",
                linewidths=2.0,
                zorder=5,
            )

        special_handles = [
            Line2D([0], [0], marker="X", color="black", linestyle="None", markersize=10,
                   markerfacecolor="none", markeredgewidth=2, label="base model"),
            Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=14,
                   markerfacecolor="none", markeredgewidth=2, label="best model"),
            Line2D([0], [0], marker="P", color="black", linestyle="None", markersize=12,
                   markerfacecolor="none", markeredgewidth=2, label="base & best"),
        ]

    title_dataset = dataset_name if dataset_name is not None else "All datasets"
    readable_title = title_suffix if title_suffix is not None else color_by
    ax.set_title(f"{title_dataset}: acc-F1 scatter by {readable_title}", fontsize=13)
    ax.set_xlabel("Best dev accuracy")
    ax.set_ylabel("Best dev F1")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.25)

    color_handles, color_labels = ax.get_legend_handles_labels()
    color_legend = legend_ax.legend(color_handles, color_labels, title=color_by, loc="upper left", frameon=True)
    legend_ax.add_artist(color_legend)

    if annotate_special and special_handles:
        legend_ax.legend(handles=special_handles, title="special markers", loc="lower left", frameon=True)

    save_or_show(fig, save_path=save_path)


def plot_acc_f1_scatter_grid(
    data: pd.DataFrame,
    color_cols: Sequence[str],
    output_dir: Optional[PathLike] = None,
    x_col: str = "best_dev_acc",
    y_col: str = "best_dev_f1",
    skip_single_value: bool = True,
):
    """Draw scatter plots for each dataset and each color column."""
    datasets = sorted(data["dataset"].astype(str).unique()) if "dataset" in data.columns else [None]

    for color_col in color_cols:
        if color_col not in data.columns:
            print(f"skip {color_col}: missing column")
            continue
        if skip_single_value and not has_multiple_values(data, color_col):
            print(f"skip {color_col}: only one value in whole dataframe")
            continue

        for dataset_name in datasets:
            sub = data[data["dataset"].astype(str) == str(dataset_name)] if dataset_name is not None else data
            if skip_single_value and not has_multiple_values(sub, color_col):
                print(f"skip {dataset_name} / {color_col}: only one value")
                continue

            save_path = None
            if output_dir is not None:
                safe_dataset = str(dataset_name).replace("/", "_").replace(" ", "_")
                safe_col = str(color_col).replace("/", "_").replace(" ", "_")
                save_path = Path(output_dir) / f"{safe_dataset}_{safe_col}_scatter.png"

            plot_acc_f1_scatter(
                data,
                color_by=color_col,
                dataset_name=dataset_name,
                x_col=x_col,
                y_col=y_col,
                save_path=save_path,
            )


# =========================
# Bar plots
# =========================


def plot_top_k_models(
    df: pd.DataFrame,
    metric: str = "best_dev_f1",
    top_k: int = 10,
    x_min="auto",
    x_margin: float = 0.005,
    output_dir: Optional[PathLike] = None,
):
    """Top-k model performance bar plot for each dataset."""
    for dataset_name, sub in df.groupby("dataset"):
        sub = sub.copy().dropna(subset=[metric])
        sub = sub.sort_values(metric, ascending=False).head(top_k)
        if len(sub) == 0:
            continue

        labels = _get_label_series(sub)
        fig, ax = plt.subplots(figsize=(9, 6))
        y_pos = np.arange(len(sub))

        ax.barh(y_pos, sub[metric])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(metric)
        ax.set_title(f"{dataset_name}: top {top_k} models by {metric}")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))

        left, right = _safe_xlim_from_values(sub[metric], x_min=x_min, x_margin=x_margin)
        ax.set_xlim(left, right)

        for i, value in enumerate(sub[metric]):
            ax.text(value, i, f" {value:.4f}", va="center", fontsize=8)

        ax.grid(axis="x", alpha=0.25)
        if left > 0:
            ax.text(
                0.99,
                -0.08,
                f"Note: x-axis starts at {left:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )

        save_path = None
        if output_dir is not None:
            save_path = Path(output_dir) / f"{dataset_name}_top_{top_k}_{metric}.png"
        save_or_show(fig, save_path=save_path)


def plot_top_k_models_by_finetune(
    df: pd.DataFrame,
    metric: str = "best_dev_f1",
    top_k: int = 10,
    x_min="auto",
    x_margin: float = 0.005,
    output_dir: Optional[PathLike] = None,
):
    """Top-k bar plot separated by dataset and fine-tune mode."""
    for dataset_name, dataset_sub in df.groupby("dataset"):
        for ft_name, sub in dataset_sub.groupby("fine_tune_type"):
            sub = sub.copy().dropna(subset=[metric])
            if len(sub) == 0:
                continue
            sub = sub.sort_values(metric, ascending=False).head(top_k)
            labels = _get_label_series(sub)

            fig, ax = plt.subplots(figsize=(9, 6))
            y_pos = np.arange(len(sub))
            ax.barh(y_pos, sub[metric])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel(metric)
            ax.set_title(f"{dataset_name}: top {top_k} models by {metric} ({ft_name})")
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))

            left, right = _safe_xlim_from_values(sub[metric], x_min=x_min, x_margin=x_margin)
            ax.set_xlim(left, right)

            for i, value in enumerate(sub[metric]):
                ax.text(value, i, f" {value:.4f}", va="center", fontsize=8)

            ax.grid(axis="x", alpha=0.25)
            if left > 0:
                ax.text(
                    0.99,
                    -0.08,
                    f"Note: x-axis starts at {left:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                )

            save_path = None
            if output_dir is not None:
                safe_ft = str(ft_name).replace("/", "_").replace(" ", "_")
                save_path = Path(output_dir) / f"{dataset_name}_{safe_ft}_top_{top_k}_{metric}.png"
            save_or_show(fig, save_path=save_path)


def plot_delta_from_base(
    df: pd.DataFrame,
    metric: str = "best_dev_f1",
    top_k: int = 15,
    base_agg: str = "max",
    x_margin: float = 0.002,
    output_dir: Optional[PathLike] = None,
):
    """Bar plot of performance delta from the base model."""
    for dataset_name, sub in df.groupby("dataset"):
        sub = sub.copy()
        base_rows = sub[sub.get("is_base_model", False)]
        if len(base_rows) == 0:
            print(f"{dataset_name}: no base model rows")
            continue

        if base_agg == "max":
            base_score = base_rows[metric].max()
        elif base_agg == "mean":
            base_score = base_rows[metric].mean()
        else:
            raise ValueError("base_agg must be 'max' or 'mean'")

        sub["delta_from_base"] = sub[metric] - base_score
        sub = sub.dropna(subset=["delta_from_base"])
        sub = sub.sort_values("delta_from_base", ascending=False).head(top_k)
        labels = _get_label_series(sub)

        fig, ax = plt.subplots(figsize=(9, 6))
        y_pos = np.arange(len(sub))
        ax.barh(y_pos, sub["delta_from_base"])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel(f"Δ {metric} from {base_agg} base model")
        ax.set_title(f"{dataset_name}: improvement over base model")

        max_abs = max(abs(sub["delta_from_base"].min()), abs(sub["delta_from_base"].max()))
        ax.set_xlim(-max_abs - x_margin, max_abs + x_margin)
        ax.grid(axis="x", alpha=0.25)

        save_path = None
        if output_dir is not None:
            save_path = Path(output_dir) / f"{dataset_name}_delta_from_base_{metric}.png"
        save_or_show(fig, save_path=save_path)


# =========================
# Sweep / box / gap plots
# =========================


def plot_sweep_line_by_finetune(
    df: pd.DataFrame,
    hyper_col: str,
    metric: str = "best_dev_f1",
    aggfunc: str = "max",
    output_dir: Optional[PathLike] = None,
):
    """Sweep line plot. Fine-tune modes are drawn as separate lines. Single-value params are skipped."""
    for dataset_name, sub in df.groupby("dataset"):
        sub = sub.copy()
        if hyper_col not in sub.columns:
            print(f"skip {dataset_name} / {hyper_col}: missing column")
            continue

        sub[hyper_col] = pd.to_numeric(sub[hyper_col], errors="coerce")
        sub = sub.dropna(subset=[hyper_col, metric, "fine_tune_type"])
        if sub[hyper_col].nunique() < 2:
            print(f"skip {dataset_name} / {hyper_col}: only one value")
            continue

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        plotted = False

        for ft_name, ft_sub in sub.groupby("fine_tune_type"):
            if ft_sub[hyper_col].nunique() < 2:
                continue

            if aggfunc == "max":
                grouped = ft_sub.groupby(hyper_col)[metric].max().reset_index().sort_values(hyper_col)
            elif aggfunc == "mean":
                grouped = ft_sub.groupby(hyper_col)[metric].mean().reset_index().sort_values(hyper_col)
            else:
                raise ValueError("aggfunc must be 'max' or 'mean'")

            ax.plot(grouped[hyper_col], grouped[metric], marker="o", label=ft_name)
            plotted = True

        if not plotted:
            plt.close(fig)
            print(f"skip {dataset_name} / {hyper_col}: no fine-tune group has multiple values")
            continue

        ax.set_xlabel(hyper_col)
        ax.set_ylabel(metric)
        ax.set_title(f"{dataset_name}: {metric} by {hyper_col}")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, alpha=0.25)
        ax.legend(title="fine-tune mode")

        save_path = None
        if output_dir is not None:
            save_path = Path(output_dir) / f"{dataset_name}_{hyper_col}_{metric}_sweep.png"
        save_or_show(fig, save_path=save_path)


def plot_box_by_option_and_finetune(
    df: pd.DataFrame,
    option_col: str,
    metric: str = "best_dev_f1",
    output_dir: Optional[PathLike] = None,
):
    """Box plot for categorical options, split by fine-tune mode. Single-value options are skipped."""
    for dataset_name, sub in df.groupby("dataset"):
        sub = sub.copy()
        if option_col not in sub.columns:
            print(f"skip {dataset_name} / {option_col}: missing column")
            continue

        sub = sub.dropna(subset=[metric, option_col, "fine_tune_type"])
        if sub[option_col].astype(str).nunique() < 2:
            print(f"skip {dataset_name} / {option_col}: only one value")
            continue

        labels, values = [], []
        for option_value in sorted(sub[option_col].astype(str).unique()):
            for ft_name in sorted(sub["fine_tune_type"].astype(str).unique()):
                group = sub[
                    (sub[option_col].astype(str) == option_value)
                    & (sub["fine_tune_type"].astype(str) == ft_name)
                ]
                if len(group) == 0:
                    continue
                labels.append(f"{option_value}\n{ft_name}")
                values.append(group[metric].values)

        if len(values) < 2:
            print(f"skip {dataset_name} / {option_col}: not enough groups")
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5.5))
        ax.boxplot(values, labels=labels, showmeans=True)
        ax.set_xlabel(option_col)
        ax.set_ylabel(metric)
        ax.set_title(f"{dataset_name}: {metric} by {option_col} and fine-tune mode")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=35, ha="right")

        save_path = None
        if output_dir is not None:
            save_path = Path(output_dir) / f"{dataset_name}_{option_col}_{metric}_box.png"
        save_or_show(fig, save_path=save_path)


def plot_acc_f1_gap(df: pd.DataFrame, top_k: int = 15, output_dir: Optional[PathLike] = None):
    """Bar plot showing models with the largest |accuracy - F1| gap."""
    if "acc_f1_gap" not in df.columns:
        if "best_dev_acc" not in df.columns or "best_dev_f1" not in df.columns:
            raise ValueError("Need best_dev_acc and best_dev_f1 columns.")
        df = df.copy()
        df["acc_f1_gap"] = df["best_dev_acc"] - df["best_dev_f1"]

    for dataset_name, sub in df.groupby("dataset"):
        sub = sub.copy().dropna(subset=["acc_f1_gap"])
        sub = sub.reindex(sub["acc_f1_gap"].abs().sort_values(ascending=False).index).head(top_k)
        labels = _get_label_series(sub)

        fig, ax = plt.subplots(figsize=(9, 6))
        y_pos = np.arange(len(sub))
        ax.barh(y_pos, sub["acc_f1_gap"])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("best_dev_acc - best_dev_f1")
        ax.set_title(f"{dataset_name}: accuracy-F1 gap")
        ax.grid(axis="x", alpha=0.25)

        save_path = None
        if output_dir is not None:
            save_path = Path(output_dir) / f"{dataset_name}_acc_f1_gap.png"
        save_or_show(fig, save_path=save_path)


def run_default_visualizations(
    df: pd.DataFrame,
    output_dir: Optional[PathLike] = None,
    metric: str = "best_dev_f1",
):
    """Run a reasonable default set of visualizations. Single-value params are skipped."""
    scatter_cols = [
        "model_role",
        "fine_tune_type",
        "experiment_group",
        "is_hyper_sweep",
        "rdrop_use",
        "warmup_use",
        "scheduler_use",
        "classifier_head",
        "pooling_config_cat",
    ]

    plot_acc_f1_scatter_grid(df, scatter_cols, output_dir=output_dir, skip_single_value=True)
    plot_top_k_models_by_finetune(df, metric=metric, top_k=10, output_dir=output_dir)
    plot_delta_from_base(df, metric=metric, top_k=15, output_dir=output_dir)

    for hyper_col in ["lr", "hidden_dropout_prob", "rdrop_alpha", "warmup_ratio", "weight_decay", "max_grad_norm"]:
        plot_sweep_line_by_finetune(df, hyper_col, metric=metric, output_dir=output_dir)

    for option_col in [
        "rdrop_use",
        "warmup_use",
        "scheduler_use",
        "pooling_config_cat",
        "classifier_head",
        "weight_decay_use",
        "grad_clip_use",
    ]:
        plot_box_by_option_and_finetune(df, option_col, metric=metric, output_dir=output_dir)

    plot_acc_f1_gap(df, output_dir=output_dir)



#===============================================
#===============================================

def add_experiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interpretable experiment scores.

    stabilization_score:
      +1 weight decay
      +1 gradient clipping
      +1 scheduler/warmup
      +1 R-Drop

    architecture_score:
      +1 MLP classifier head
      +1 last_mean pooling

    balanced_score:
      (best_dev_acc + best_dev_f1) / 2
    """
    df = df.copy()

    def is_use_col(col):
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].astype(str).str.lower().eq("use")

    # Stabilization / regularization score
    df["stabilization_score"] = 0
    df["stabilization_score"] += is_use_col("weight_decay_use").astype(int)
    df["stabilization_score"] += is_use_col("grad_clip_use").astype(int)
    df["stabilization_score"] += is_use_col("scheduler_use").astype(int)
    df["stabilization_score"] += is_use_col("rdrop_use").astype(int)

    # Classifier head label
    if "use_simple_classifier" in df.columns:
        simple = df["use_simple_classifier"].astype(str).str.lower()
        df["classifier_head"] = np.select(
            [simple.eq("true"), simple.eq("false")],
            ["simple head", "MLP head"],
            default="unknown",
        )
    else:
        df["classifier_head"] = "unknown"

    # Architecture score
    df["architecture_score"] = 0
    df["architecture_score"] += df["classifier_head"].eq("MLP head").astype(int)

    if "pooling_config" in df.columns:
        pooling = clean_category(df["pooling_config"], default="last").str.lower()
    elif "pooling_config_cat" in df.columns:
        pooling = clean_category(df["pooling_config_cat"], default="last").str.lower()
    else:
        pooling = pd.Series(["last"] * len(df), index=df.index)

    df["pooling_family"] = pooling
    df["architecture_score"] += pooling.eq("last_mean").astype(int)

    # Mean pooling alone is better treated as a separate representation variant.
    df["uses_mean_only_pooling"] = pooling.eq("mean")

    if "best_dev_acc" in df.columns and "best_dev_f1" in df.columns:
        df["balanced_score"] = (df["best_dev_acc"] + df["best_dev_f1"]) / 2

    df["total_upgrade_score"] = df["stabilization_score"] + df["architecture_score"]

    return df


def plot_upgrade_score_heatmap(
    df: pd.DataFrame,
    metric: str = "balanced_score",
    aggfunc: str = "max",
    exclude_mean_only_pooling: bool = True,
    output_dir: Optional[PathLike] = None,
):
    """
    Heatmap of performance by stabilization score and architecture score.

    Rows: stabilization_score
    Columns: architecture_score
    Cell: aggregated metric
    """
    df = add_experiment_scores(df)

    if exclude_mean_only_pooling and "uses_mean_only_pooling" in df.columns:
        df = df[~df["uses_mean_only_pooling"]].copy()

    for dataset_name, sub in df.groupby("dataset"):
        sub = sub.dropna(subset=[metric, "stabilization_score", "architecture_score"]).copy()
        if len(sub) == 0:
            continue

        pivot = sub.pivot_table(
            index="stabilization_score",
            columns="architecture_score",
            values=metric,
            aggfunc=aggfunc,
        ).sort_index(ascending=True)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.imshow(pivot.values, aspect="auto")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([str(x) for x in pivot.columns])
        ax.set_yticklabels([str(y) for y in pivot.index])

        ax.set_xlabel("Architecture score")
        ax.set_ylabel("Stabilization score")
        ax.set_title(f"{dataset_name}: {aggfunc} {metric} by upgrade scores")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(metric)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.4f}", ha="center", va="center", fontsize=9)

        save_path = None
        if output_dir is not None:
            save_path = Path(output_dir) / f"{dataset_name}_upgrade_score_heatmap_{metric}_{aggfunc}.png"

        save_or_show(fig, save_path=save_path)
        
        
        
def plot_score_vs_metric(
    df: pd.DataFrame,
    score_col: str = "stabilization_score",
    metric: str = "balanced_score",
    output_dir: Optional[PathLike] = None,
):
    """
    Scatter/mean plot for score vs metric.
    Useful for checking whether stronger stabilization/architecture tends to improve performance.
    """
    df = add_experiment_scores(df)

    for dataset_name, sub in df.groupby("dataset"):
        sub = sub.dropna(subset=[score_col, metric]).copy()
        if len(sub) == 0:
            continue

        grouped = (
            sub.groupby(score_col)[metric]
            .agg(["mean", "max", "std", "count"])
            .reset_index()
            .sort_values(score_col)
        )

        fig, ax = plt.subplots(figsize=(7.5, 5.5))

        # Scatter all runs
        jitter = np.random.default_rng(42).normal(0, 0.04, size=len(sub))
        ax.scatter(
            sub[score_col] + jitter,
            sub[metric],
            alpha=0.45,
            s=45,
            edgecolors="black",
            linewidths=0.3,
            label="individual runs",
        )

        # Mean trend
        ax.plot(
            grouped[score_col],
            grouped["mean"],
            marker="o",
            linewidth=2,
            label="mean",
        )

        # Max trend
        ax.plot(
            grouped[score_col],
            grouped["max"],
            marker="s",
            linewidth=2,
            linestyle="--",
            label="max",
        )

        ax.set_xlabel(score_col)
        ax.set_ylabel(metric)
        ax.set_title(f"{dataset_name}: {metric} by {score_col}")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, alpha=0.25)
        ax.legend()

        save_path = None
        if output_dir is not None:
            save_path = Path(output_dir) / f"{dataset_name}_{score_col}_vs_{metric}.png"

        save_or_show(fig, save_path=save_path)
        
def run_plus_visualizations(
    df: pd.DataFrame,
    output_dir: Optional[PathLike] = None,
):
        # Upgrade score visualizations
    scored_df = add_experiment_scores(df)

    for metric_name in ["best_dev_f1", "balanced_score"]:
        if metric_name in scored_df.columns:
            plot_score_vs_metric(
                scored_df,
                score_col="stabilization_score",
                metric=metric_name,
                output_dir=output_dir,
            )
            plot_score_vs_metric(
                scored_df,
                score_col="architecture_score",
                metric=metric_name,
                output_dir=output_dir,
            )
            plot_upgrade_score_heatmap(
                scored_df,
                metric=metric_name,
                aggfunc="max",
                output_dir=output_dir,
            )
            plot_upgrade_score_heatmap(
                scored_df,
                metric=metric_name,
                aggfunc="mean",
                output_dir=output_dir,
            )