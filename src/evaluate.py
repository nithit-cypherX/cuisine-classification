"""
evaluate.py
-----------
Computes all evaluation metrics across all 12 conditions.

Experiment matrix evaluated:
  3 embedding types (bow, sparse, dense) × 4 k values (0, 4, 8, 16)
  = 12 total conditions

Metrics computed per condition:
  - Accuracy
  - Macro F1
  - Per-class F1
  - Invalid output rate

Figures saved to results/figures/:
  - summary_accuracy.png        ← Accuracy across all conditions
  - summary_macro_f1.png        ← Macro F1 across all conditions
  - heatmap_macro_f1.png        ← Heatmap: embedding type vs k value
  - per_class_f1_{emb}.png      ← Per-class F1 per embedding type
  - confusion_matrix_{cond}.png ← Confusion matrix per condition

Run after inference.py:
  python src/evaluate.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from config import (
    PREDICTIONS_CSV, RESULTS_FIG_DIR,
    TARGET_CUISINES, EMBEDDING_TYPES, K_VALUES
)

matplotlib.use("Agg")   # Non-interactive backend for script use

DISPLAY_LABELS = [c.replace("_", " ").title() for c in TARGET_CUISINES]
EMB_COLORS     = {"bow": "#3266ad", "sparse": "#1D9E75", "dense": "#D85A30"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"[evaluate] Loaded predictions: {len(df)} recipes, {len(df.columns)} columns")
    return df


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute accuracy, macro F1, and invalid rate for one condition."""
    invalid_mask = y_pred == "invalid"
    invalid_rate = invalid_mask.mean()
    valid_true   = y_true[~invalid_mask]
    valid_pred   = y_pred[~invalid_mask]

    if len(valid_pred) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "invalid_rate": 1.0}

    acc = accuracy_score(valid_true, valid_pred)
    f1  = f1_score(valid_true, valid_pred,
                   average="macro", labels=TARGET_CUISINES, zero_division=0)
    return {
        "accuracy":     round(acc, 4),
        "macro_f1":     round(f1, 4),
        "invalid_rate": round(invalid_rate, 4)
    }


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build summary results table for all 12 conditions."""
    rows = []
    for emb_type in EMBEDDING_TYPES:
        for k in K_VALUES:
            condition = f"{emb_type}_k{k}"
            pred_col  = f"{condition}_pred"
            metrics   = compute_metrics(df["cuisine"], df[pred_col])
            rows.append({
                "Condition":    condition,
                "Embedding":    emb_type.upper(),
                "k":            k,
                "Accuracy":     metrics["accuracy"],
                "Macro F1":     metrics["macro_f1"],
                "Invalid rate": f"{metrics['invalid_rate']:.1%}"
            })
    summary = pd.DataFrame(rows)
    print("\n── Summary Results (all 12 conditions) ─────────────────")
    print(summary.to_string(index=False))
    summary.to_csv(f"{RESULTS_FIG_DIR}/summary_results.csv", index=False)
    return summary


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_summary_bars(summary: pd.DataFrame, metric: str, title: str, filename: str) -> None:
    """Bar chart comparing a metric across all conditions grouped by embedding type."""
    os.makedirs(RESULTS_FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    x       = np.arange(len(K_VALUES))
    width   = 0.25
    offsets = [-1, 0, 1]

    for i, emb_type in enumerate(EMBEDDING_TYPES):
        subset = summary[summary["Embedding"] == emb_type.upper()][metric].values
        bars   = ax.bar(x + offsets[i] * width, subset, width,
                        label=emb_type.upper(), color=EMB_COLORS[emb_type], alpha=0.85)
        for bar, val in zip(bars, subset):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in K_VALUES])
    ax.set_ylabel(metric)
    ax.set_title(title, fontsize=13)
    ax.legend(title="Embedding")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    save_path = f"{RESULTS_FIG_DIR}/{filename}"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] Saved → {save_path}")


def plot_heatmap(summary: pd.DataFrame) -> None:
    """Heatmap of Macro F1 with embedding type as rows and k as columns."""
    pivot = summary.pivot(index="Embedding", columns="k", values="Macro F1")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGn",
        linewidths=0.5, ax=ax,
        vmin=0, vmax=1,
        cbar_kws={"label": "Macro F1"}
    )
    ax.set_title("Macro F1 Heatmap: Embedding Type × k Value", fontsize=13)
    ax.set_xlabel("k (number of retrieved examples)")
    ax.set_ylabel("Embedding Type")
    plt.tight_layout()
    save_path = f"{RESULTS_FIG_DIR}/heatmap_macro_f1.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] Saved → {save_path}")


def plot_per_class_f1(df: pd.DataFrame) -> None:
    """Per-class F1 bar chart for each embedding type (across k values)."""
    for emb_type in EMBEDDING_TYPES:
        fig, axes = plt.subplots(1, len(K_VALUES), figsize=(14, 4), sharey=True)
        fig.suptitle(f"Per-class F1 — {emb_type.upper()} Embedding", fontsize=13)

        for ax, k in zip(axes, K_VALUES):
            pred_col  = f"{emb_type}_k{k}_pred"
            valid     = df[df[pred_col] != "invalid"]
            f1_values = f1_score(
                valid["cuisine"], valid[pred_col],
                average=None, labels=TARGET_CUISINES, zero_division=0
            )
            ax.bar(range(len(TARGET_CUISINES)), f1_values,
                   color=EMB_COLORS[emb_type], alpha=0.85)
            ax.set_xticks(range(len(TARGET_CUISINES)))
            ax.set_xticklabels(DISPLAY_LABELS, rotation=35, ha="right", fontsize=8)
            ax.set_title(f"k={k}", fontsize=11)
            ax.set_ylim(0, 1.05)
            if k == 0:
                ax.set_ylabel("F1 Score")

        plt.tight_layout()
        save_path = f"{RESULTS_FIG_DIR}/per_class_f1_{emb_type}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[evaluate] Saved → {save_path}")


def plot_confusion_matrices(df: pd.DataFrame) -> None:
    """Confusion matrix for every condition (12 total)."""
    for emb_type in EMBEDDING_TYPES:
        for k in K_VALUES:
            condition = f"{emb_type}_k{k}"
            pred_col  = f"{condition}_pred"
            valid     = df[df[pred_col] != "invalid"]
            cm        = confusion_matrix(valid["cuisine"], valid[pred_col],
                                        labels=TARGET_CUISINES)

            fig, ax = plt.subplots(figsize=(7, 6))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=DISPLAY_LABELS
            )
            disp.plot(ax=ax, colorbar=True, cmap="Blues")
            ax.set_title(f"Confusion Matrix | {emb_type.upper()} | k={k}", fontsize=12, pad=10)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            save_path = f"{RESULTS_FIG_DIR}/confusion_matrix_{condition}.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"[evaluate] Saved → {save_path}")


# ── Per-class report ──────────────────────────────────────────────────────────

def print_per_class_reports(df: pd.DataFrame) -> None:
    """Print full classification report for each condition."""
    for emb_type in EMBEDDING_TYPES:
        for k in K_VALUES:
            condition = f"{emb_type}_k{k}"
            pred_col  = f"{condition}_pred"
            valid     = df[df[pred_col] != "invalid"]
            print(f"\n── Classification Report | {emb_type.upper()} | k={k} ──────────")
            print(classification_report(
                valid["cuisine"], valid[pred_col],
                labels=TARGET_CUISINES, zero_division=0
            ))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS_FIG_DIR, exist_ok=True)
    df = load_predictions()

    summary = build_summary_table(df)
    print_per_class_reports(df)

    plot_summary_bars(summary, "Accuracy", "Accuracy: All Conditions", "summary_accuracy.png")
    plot_summary_bars(summary, "Macro F1", "Macro F1: All Conditions", "summary_macro_f1.png")
    plot_heatmap(summary)
    plot_per_class_f1(df)
    plot_confusion_matrices(df)

    print("\n[evaluate] All evaluation complete.")
    print(f"[evaluate] Results saved to: {RESULTS_FIG_DIR}/")
