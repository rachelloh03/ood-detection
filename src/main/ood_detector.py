"""
OOD Detector class.
"""

from collections.abc import Callable
from typing import Literal
from main.transformations import Transformations
import torch
from torch.utils.data import DataLoader


class OODDetector:
    """
    OOD Detector class.

    Refer to docs/ood_detection.md for more details, under OOD Detector Class section.
    """

    def __init__(
        self,
        embedding_function: Transformations,
        scoring_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        id_train_data: torch.Tensor,
    ):
        """
        Refer to docs/ood_detection.md for more details, under OOD Detector Class section.
        """
        self.embedding_function = embedding_function
        self.scoring_function = scoring_function
        self.id_train_data = id_train_data  # (N, D)
        self.embedding_function.fit(id_train_data)
        self.embedded_id_train_data = self.embedding_function(
            id_train_data
        )  # (N, embedding_dim)

    def evaluate(
        self,
        id_test_data: DataLoader | torch.Tensor,  # (A, D)
        ood_test_data: DataLoader | torch.Tensor,  # (B, D)
        threshold: float | None = None,  # If None, it is inferred from the data.
        threshold_type: Literal["value", "percentile"] = "value",
    ):
        """
        Evaluate the OOD detector on the ID and OOD test data.

        Args:
            id_test_data: ID test data. Either a DataLoader or a tensor, has shape (A, D).
            ood_test_data: OOD test data. Either a DataLoader or a tensor, has shape (B, D).
            threshold: Threshold for the OOD detector.
            threshold_type: Type of threshold. "value" means the threshold is a fixed value. =
            "percentile" means the threshold is the percentile of the OOD scores.

        Examples:
            1. If the expected contamination rate is 5%, we can set threshold_type to "percentile"
                and threshold to 0.95, representing 95th percentile of OOD scores.
            2. If we want to set the threshold to a fixed value of 10,
                we can set threshold_type to "value" and threshold to 10.
            3. If threshold_type is "percentile" and the threshold is not set, it is assumed to be A/(A+B),
                representing A ID samples / (A ID samples + B OOD samples).

        Returns:
            confusion_matrix: Confusion matrix. Has shape (2, 2).
            true_positive_rate: True positive rate.
            false_positive_rate: False positive rate.
        """
        if isinstance(id_test_data, DataLoader):
            A = len(id_test_data)
        else:
            A = id_test_data.shape[0]
        if isinstance(ood_test_data, DataLoader):
            B = len(ood_test_data)
        else:
            B = ood_test_data.shape[0]

        id_scores = self.score(id_test_data)  # (A,)
        ood_scores = self.score(ood_test_data)  # (B,)

        all_scores = torch.cat([id_scores, ood_scores])
        if threshold_type == "percentile":
            if threshold is None:
                threshold = A / (A + B)
            threshold = torch.quantile(
                all_scores,
                torch.tensor(threshold, dtype=all_scores.dtype),
                # id_scores, torch.tensor(threshold, dtype=id_scores.dtype)
                # # compute threshold only on ID test data to avoid data leakage
            )
            # print("Threshold:", threshold)

        positive_id = id_scores > threshold  # (A,)
        positive_ood = ood_scores > threshold  # (B,)

        true_negative = (~positive_id).sum()
        false_positive = (positive_id).sum()
        false_negative = (~positive_ood).sum()
        true_positive = (positive_ood).sum()
        confusion_matrix = torch.tensor(
            [[true_positive, false_negative], [false_positive, true_negative]],
            dtype=torch.long,
        )
        true_positive_rate = (
            true_positive.float() / (true_positive + false_negative).float()
        )
        false_positive_rate = (
            false_positive.float() / (false_positive + true_negative).float()
        )

        # plot for debugging purposes
        # plt.figure(figsize=(10, 4))
        # plt.subplot(1, 2, 1)
        # plt.hist(id_scores.cpu().numpy(), bins=50, alpha=0.7, label='ID')
        # plt.hist(ood_scores.cpu().numpy(), bins=50, alpha=0.7, label='OOD')
        # plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
        # plt.legend()
        # plt.xlabel('OOD Score')
        # plt.title('Score Distributions')

        # plt.subplot(1, 2, 2)
        # plt.boxplot([id_scores.cpu().numpy(), ood_scores.cpu().numpy()], labels=['ID', 'OOD'])
        # plt.axhline(threshold, color='r', linestyle='--')
        # plt.ylabel('OOD Score')
        # plt.title('Score Comparison')
        # plt.show()
        # print(f"Threshold: {threshold:.4f}")

        return confusion_matrix, true_positive_rate, false_positive_rate

    def score(
        self,
        input_data: DataLoader | torch.Tensor,
    ):
        """
        Return the OOD score for the input.

        Args:
            input_data: Input data to score. Either a DataLoader or a tensor, has shape (B, D).

        Returns:
            OOD score for the input. Has shape (B,).
        """
        input_embedding = self.embedding_function(input_data)  # (B, D)
        return self.scoring_function(
            self.embedded_id_train_data, input_embedding
        )  # (B,)
