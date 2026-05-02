"""
evaluate.py
-----------
Computes all evaluation metrics and generates result visualisations.

Metrics computed:
  - Accuracy (per strategy)
  - Macro F1 score (per strategy)
  - Per-class F1 via classification_report
  - Invalid output rate (per strategy)
  - Confusion matrix (per strategy)

Outputs saved to results/figures/:
  - confusion_matrix_zero_shot.png
  - confusion_matrix_dynamic_few_shot.png
  - per_class_f1_comparison.png

Run after inference.py:
  python src/evaluate.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from config import (
    PREDICTIONS_CSV, RESULTS_FIG_DIR,
    TARGET_CUISINES
)

STRATEGIES = {
    "Zero-shot":          "zero_shot_pred",
    "Dynamic Few-shot":   "dynamic_few_shot_pred"
}


def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"[evaluate] Loaded {len(df)} predictions.")
    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy, macro F1, and invalid rate per strategy."""
    rows = []
    for name, col in STRATEGIES.items():
        valid     = df[df[col] != "invalid"]
        invalid_n = (df[col] == "invalid").sum()
        acc  = accuracy_score(valid["cuisine"], valid[col])
        f1   = f1_score(valid["cuisine"], valid[col],
                        average="macro", labels=TARGET_CUISINES, zero_division=0)
        rows.append({
            "Strategy":        name,
            "Accuracy":        round(acc, 4),
            "Macro F1":        round(f1, 4),
            "Invalid outputs": f"{invalid_n} ({invalid_n/len(df):.1%})"
        })
    summary = pd.DataFrame(rows)
    print("\n── Summary Results ──────────────────────────────")
    print(summary.to_string(index=False))
    return summary


def compute_per_class(df: pd.DataFrame) -> None:
    """Print per-class classification report for each strategy."""
    for name, col in STRATEGIES.items():
        valid = df[df[col] != "invalid"]
        print(f"\n── Per-class Report: {name} ─────────────────────")
        print(classification_report(
            valid["cuisine"], valid[col],
            labels=TARGET_CUISINES, zero_division=0
        ))


def plot_confusion_matrices(df: pd.DataFrame) -> None:
    """Generate and save a confusion matrix for each strategy."""
    os.makedirs(RESULTS_FIG_DIR, exist_ok=True)
    labels = TARGET_CUISINES
    display_labels = [c.replace("_", " ").title() for c in labels]

    for name, col in STRATEGIES.items():
        valid = df[df[col] != "invalid"]
        cm = confusion_matrix(valid["cuisine"], valid[col], labels=labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=ax, colorbar=True, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {name}", fontsize=13, pad=12)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        filename = name.lower().replace(" ", "_").replace("-", "_")
        save_path = f"{RESULTS_FIG_DIR}/confusion_matrix_{filename}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[evaluate] Saved → {save_path}")


def plot_per_class_f1(df: pd.DataFrame) -> None:
    """Bar chart comparing per-class F1 across strategies."""
    os.makedirs(RESULTS_FIG_DIR, exist_ok=True)

    f1_data = {}
    for name, col in STRATEGIES.items():
        valid = df[df[col] != "invalid"]
        f1_per_class = f1_score(
            valid["cuisine"], valid[col],
            average=None, labels=TARGET_CUISINES, zero_division=0
        )
        f1_data[name] = f1_per_class

    fig, ax = plt.subplots(figsize=(10, 5))
    x       = range(len(TARGET_CUISINES))
    width   = 0.35
    colors  = ["#3266ad", "#1D9E75"]

    for idx, (name, values) in enumerate(f1_data.items()):
        offset = (idx - 0.5) * width
        ax.bar([xi + offset for xi in x], values, width, label=name, color=colors[idx], alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels([c.replace("_", " ").title() for c in TARGET_CUISINES], rotation=20, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-class F1 Score: Zero-shot vs Dynamic Few-shot", fontsize=13)
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    save_path = f"{RESULTS_FIG_DIR}/per_class_f1_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] Saved → {save_path}")


if __name__ == "__main__":
    df = load_predictions()
    compute_summary(df)
    compute_per_class(df)
    plot_confusion_matrices(df)
    plot_per_class_f1(df)
    print("\n[evaluate] All evaluation complete.")
