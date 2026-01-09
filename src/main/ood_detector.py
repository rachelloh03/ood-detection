"""
OOD Detector class.
"""

from collections.abc import Callable
from typing import Literal
from src.main.transformations import Transformations
import torch


# NOT TESTED YET
class OODDetector:
    """
    OOD Detector class.

    Many post-hoc OOD detection methods are of the following form:
    1) Find a embedding function h() that maps each prompt x to a vector representation h(x).
        An example of h(x) is the hidden state of a model given input x.
    2) Estimate the distribution of h(x) on a training set.
    3) For a new prompt x', compute h(x')
        and determine if it is an outlier of the distribution via a scoring mechanism.
        An example is taking the Mahalanobis distance of h(x') to the ID distribution of h(x).
    4) Set a threshold on the score and determine if x' is OOD.
    """

    def __init__(
        self,
        embedding_function: Transformations,
        scoring_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        id_train_data: torch.Tensor,
    ):
        """
        Args:
            embedding_function: Transformations object that maps each prompt x to a vector representation h(x).
                Why it's an object is to allow for things like PCA that are fitted on the ID training data.
            scoring_function: Callable that takes (ID embeddings, new input embeddings), and returns an OOD score.
                (An example is the Mahalanobis distance of the new input embedding to the ID distribution.)
                If ID embeddings has shape (N, D) and new input embeddings has shape (M, D),
                then scoring_function should have shape (M,), one score for each new input embedding.
            id_train_data: Training data for the ID distribution.
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
        id_scores = self.score(id_test_data)  # (A,)
        ood_scores = self.score(ood_test_data)  # (B,)

        all_scores = torch.cat([id_scores, ood_scores])
        if threshold_type == "percentile":
            if threshold is None:
                threshold = A / (A + B)
            threshold = torch.quantile(all_scores, threshold)

        positive_id = id_scores > threshold  # (A,)
        positive_ood = ood_scores > threshold  # (B,)

        true_negative = (~positive_id).sum()
        false_positive = (positive_id).sum()
        false_negative = (~positive_ood).sum()
        true_positive = (positive_ood).sum()
        confusion_matrix = torch.tensor(
            [[true_negative, false_positive], [false_negative, true_positive]],
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
