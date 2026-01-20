"""Plots AUROC curve for OOD detection."""

from main.ood_detector import OODDetector
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import auc


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

    # calculate thresholds based on range of ID and OOD scores
    # id_scores = ood_detector.score(id_data)
    # ood_scores = ood_detector.score(ood_data)
    # all_scores = torch.cat([id_scores, ood_scores])
    # thresholds = torch.linspace(
    #     all_scores.min() - 0.1,  # Start slightly below minimum
    #     all_scores.max() + 0.1,  # End slightly above maximum
    #     100
    # )
    # print(thresholds)

    roc_curve = []
    for threshold in thresholds:
        _, tpr, fpr = ood_detector.evaluate(
            id_data, ood_data, threshold, threshold_type="percentile"
        )
        # _, tpr, fpr = ood_detector.evaluate(
        #     id_data, ood_data, threshold.item(), threshold_type="value"
        # )
        roc_curve.append((fpr, tpr))
    roc_curve = np.array(roc_curve)
    roc_curve = roc_curve[np.argsort(roc_curve[:, 0])]
    auroc = auc(roc_curve[:, 0], roc_curve[:, 1])
    plt.plot(roc_curve[:, 0], roc_curve[:, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(save_path)
    plt.show()
    plt.close()
    return auroc
