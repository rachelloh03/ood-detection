"""
OOD Detector class.
"""

from collections.abc import Callable
from typing import Literal
from main.transformations import Transformations
import torch


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

        assert self.embedded_id_train_data.shape[0] == id_train_data.shape[0]

    def evaluate(
        self,
        id_test_data: torch.Tensor,  # (A, D)
        ood_test_data: torch.Tensor,  # (B, D)
        threshold: float | None = None,  # If None, it is inferred from the data.
        threshold_type: Literal["value", "percentile"] = "value",
    ):
        """
        Evaluate the OOD detector on the ID and OOD test data.

        Args:
            id_test_data: ID test data. Has shape (A, D).
            ood_test_data: OOD test data. Has shape (B, D).
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
        assert (
            id_test_data.shape[1] == ood_test_data.shape[1]
        ), "ID and OOD test data must have the same number of dimensions"

        A, B = id_test_data.shape[0], ood_test_data.shape[0]
        if A == 0 or B == 0:
            raise ValueError("ID and OOD test data must have at least one sample")
        id_scores = self.score(id_test_data)  # (A,)
        ood_scores = self.score(ood_test_data)  # (B,)

        all_scores = torch.cat([id_scores, ood_scores])
        if threshold_type == "percentile":
            if threshold is None:
                threshold = A / (A + B)
            threshold = torch.quantile(
                all_scores, torch.tensor(threshold, dtype=all_scores.dtype)
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
        return confusion_matrix, true_positive_rate, false_positive_rate

    def score(
        self,
        input_data: torch.Tensor,
    ):
        """
        Return the OOD score for the input.

        Args:
            input_data: Input data to score. Has shape (B, D).

        Returns:
            OOD score for the input. Has shape (B,).
        """
        input_embedding = self.embedding_function(input_data)  # (B, D)
        return self.scoring_function(
            self.embedded_id_train_data, input_embedding
        )  # (B,)
