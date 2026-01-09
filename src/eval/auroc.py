"""Plots AUROC curve for OOD detection."""

from main.ood_detector import OODDetector
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import auc


# HAVEN'T TESTED THIS YET
def get_auroc(
    ood_detector: OODDetector,
    id_data: torch.Tensor,
    ood_data: torch.Tensor,
    save_path="auroc.png",
):
    """
    Get AUROC curve for OOD detection.

    It does this by taking OOD scores and comparing against a varying threshold,
    then plotting the points on the ROC curve from the different threshold values.

    ood_detector: function that takes in a (batched) data tensor (B, L) and returns B OOD scores.
        Higher score means more OOD.
    id_data: in-distribution data.
    ood_data: out-of-distribution data.

    Returns:
        auc: AUROC score.
    """
    thresholds = torch.linspace(0, 1, 100)
    roc_curve = []
    for threshold in thresholds:
        _, tpr, fpr = ood_detector.evaluate(
            id_data, ood_data, threshold, threshold_type="percentile"
        )
        roc_curve.append((fpr, tpr))
    roc_curve = np.array(roc_curve)
    roc_curve = roc_curve[np.argsort(roc_curve[:, 0])]
    auroc = auc(roc_curve[:, 0], roc_curve[:, 1])
    plt.plot(roc_curve[:, 0], roc_curve[:, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(save_path)
    plt.close()
    return auroc
