"""Render evaluation charts directly from measured metrics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Backend must be selected before pyplot import.
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plot_results():
    """Generate reproducible charts without reading raw applicant data."""
    report = json.loads((ROOT / "models/metrics.json").read_text())
    destination = ROOT / "docs/images"
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f4f6f5",
        }
    )
    matrix = np.asarray(report["test"]["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(matrix, cmap="Greens")
    axis.set(
        xticks=range(2),
        yticks=range(2),
        xticklabels=report["classes"],
        yticklabels=report["classes"],
        xlabel="Predicted category",
        ylabel="Actual category",
        title="Ethereum Fraud Explorer · held-out confusion matrix",
    )
    for row in range(2):
        for col in range(2):
            axis.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > matrix.max() / 2 else "#123a30",
            )
    fig.colorbar(image, ax=axis, label="Number of rows")
    fig.tight_layout()
    fig.savefig(destination / "confusion-matrix.png", dpi=150)
    plt.close(fig)
    candidates = report["validation_candidates"]
    fig, axis = plt.subplots(figsize=(8, 4))
    names = list(candidates)
    axis.barh(
        [name.replace("_", " ") for name in names],
        [candidates[name]["average_precision"] for name in names],
        color="#24664e",
    )
    axis.set(
        xlim=(0, 1),
        xlabel="Validation average precision (higher is better)",
        title="Compare candidates before opening the test results",
    )
    fig.tight_layout()
    fig.savefig(destination / "model-comparison.png", dpi=150)
    plt.close(fig)
    values = sorted(report["permutation_importance"].items(), key=lambda item: item[1])
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        [item[0] for item in values], [item[1] for item in values], color="#24664e"
    )
    axis.axvline(0, color="#718277", linewidth=1)
    axis.set(
        xlabel="Mean validation average precision drop after shuffling (3 repeats)",
        title="Permutation importance · predictive dependence, not causation",
    )
    fig.tight_layout()
    fig.savefig(destination / "feature-importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_results()
