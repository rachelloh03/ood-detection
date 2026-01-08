"""Plots AUROC curve for OOD detection."""

from collections.abc import Callable
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


# HAVEN'T TESTED THIS YET
def get_auroc(
    ood_detector: Callable,
    id_data: torch.Tensor,
    ood_data: torch.Tensor,
    score_lower_bound=0,
    score_upper_bound=1,
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
    thresholds = torch.linspace(score_lower_bound, score_upper_bound, 100)
    roc_curve = []
    for threshold in thresholds:
        is_ood = ood_detector(id_data) > threshold
        is_ood_ood = ood_detector(ood_data) > threshold
        tpr = is_ood.sum() / is_ood.sum()
        fpr = is_ood_ood.sum() / is_ood_ood.sum()
        roc_curve.append((fpr, tpr))
    roc_curve = np.array(roc_curve)
    auc = roc_auc_score(roc_curve[:, 0], roc_curve[:, 1])
    plt.plot(roc_curve[:, 0], roc_curve[:, 1])
    plt.savefig(save_path)
    plt.close()
    return auc
