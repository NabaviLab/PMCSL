from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(
    matrix,
    class_names,
    output_path,
    *,
    normalize: bool = False,
) -> None:
    """Save a confusion matrix figure."""
    matrix = np.asarray(matrix, dtype=np.float32)

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix),
            where=row_sums != 0,
        )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    display.plot(
        ax=ax,
        values_format=".2f" if normalize else "g",
        colorbar=False,
    )
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_multiclass_roc(
    targets,
    probabilities,
    class_names,
    output_path,
) -> None:
    """Save one-vs-rest ROC curves for all classes."""
    y_true = np.asarray(targets)
    y_prob = np.asarray(probabilities)
    classes = np.arange(len(class_names))
    y_binary = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(7, 6))

    for class_index, class_name in enumerate(class_names):
        RocCurveDisplay.from_predictions(
            y_binary[:, class_index],
            y_prob[:, class_index],
            name=str(class_name),
            ax=ax,
        )

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("One-vs-Rest ROC Curves")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_attention_overlay(
    image,
    heatmap,
    output_path,
    *,
    alpha: float = 0.45,
) -> None:
    """Overlay a normalized heatmap on an RGB image."""
    image = np.asarray(image)
    heatmap = np.asarray(heatmap, dtype=np.float32)

    if heatmap.shape[:2] != image.shape[:2]:
        raise ValueError("Heatmap and image spatial dimensions must match.")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(heatmap, alpha=alpha, cmap="jet")
    ax.axis("off")
    fig.tight_layout(pad=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
