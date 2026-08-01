from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_classification_metrics(
    targets: list[int],
    probabilities: list[list[float]],
) -> dict:
    """
    Compute slide-level metrics for four-class grading.
    """
    y_true = np.asarray(targets)
    y_prob = np.asarray(probabilities)
    y_pred = y_prob.argmax(axis=1)

    output = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    try:
        output["macro_auc_ovr"] = float(
            roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        output["macro_auc_ovr"] = float("nan")

    return output
