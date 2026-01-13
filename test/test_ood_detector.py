import pytest

from src.main.ood_detector import OODDetector
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.main.transformations import Transformations
from src.main.scoring_functions import mahalanobis_distance


def _create_detector():
    transformations = Transformations(
        [
            PCA(n_components=2),
            StandardScaler(),
        ]
    )
    id_train = torch.tensor([[1.0, 1.0], [1.1, 1.0], [0.9, 1.2]])
    return OODDetector(
        embedding_function=transformations,
        scoring_function=mahalanobis_distance,
        id_train_data=id_train,
    )


def test_ood_detector_score():
    detector = _create_detector()
    id_test = torch.tensor([[1.0, 1.0], [1.0, 1.1], [1.2, 0.9]])
    ood_test = torch.tensor([[3.0, 3.0], [4.0, 4.0], [1.0, 1.1], [5.0, 5.0]])

    scores = detector.score(id_test)
    print("Scores for ID test data:", scores)
    assert scores.shape[0] == id_test.shape[0]

    scores = detector.score(ood_test)
    print("Scores for OOD test data:", scores)
    assert scores.shape[0] == ood_test.shape[0]


def test_ood_detector_evaluate_value_threshold():
    detector = _create_detector()
    id_test = torch.tensor([[1.0, 1.0], [1.0, 1.1], [1.2, 0.9]])
    ood_test = torch.tensor([[3.0, 3.0], [4.0, 4.0], [1.0, 1.1], [5.0, 5.0]])

    threshold = 1  # very low, will mark some ID as OOD
    cm, tpr, fpr = detector.evaluate(
        id_test, ood_test, threshold=threshold, threshold_type="value"
    )
    assert 0 < tpr < 1
    assert 0 < fpr < 1


def test_ood_detector_evaluate_percentile_threshold_default():
    detector = _create_detector()
    id_test = torch.tensor([[1.0, 1.0], [1.0, 1.1], [1.2, 0.9]])
    ood_test = torch.tensor([[3.0, 3.0], [4.0, 4.0], [1.0, 1.1], [5.0, 5.0]])

    cm, tpr, fpr = detector.evaluate(
        id_test, ood_test, threshold_type="percentile"
    )  # by default, threshold is 3/6 = 0.5
    assert tpr == pytest.approx(3 / 4)
    assert fpr == pytest.approx(1 / 3)


def test_ood_detector_evaluate_percentile_threshold_custom():
    detector = _create_detector()
    id_test = torch.tensor([[1.0, 1.0], [1.0, 1.1], [1.2, 0.9]])
    ood_test = torch.tensor([[3.0, 3.0], [4.0, 4.0], [1.0, 1.1], [5.0, 5.0]])

    cm, tpr, fpr = detector.evaluate(
        id_test, ood_test, threshold_type="percentile", threshold=0.8
    )
    assert tpr == pytest.approx(1 / 2)
    assert fpr == 0


def test_ood_detector_evaluate_perfect_separation():
    detector = _create_detector()
    id_test = torch.tensor([[1.0, 1.0], [1.0, 1.1], [1.2, 0.9]])
    ood_test = torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])

    threshold = 10  # now it should be correctly mark only OOD as OOD
    cm, tpr, fpr = detector.evaluate(
        id_test, ood_test, threshold=threshold, threshold_type="value"
    )
    assert tpr == 1
    assert fpr == 0


if __name__ == "__main__":
    test_ood_detector_score()
    test_ood_detector_evaluate_value_threshold()
    test_ood_detector_evaluate_percentile_threshold_default()
    test_ood_detector_evaluate_percentile_threshold_custom()
    test_ood_detector_evaluate_perfect_separation()
