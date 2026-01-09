from main.ood_detector import OODDetector
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from main.transformations import Transformations
from main.scoring_functions import mahalanobis_distance


def test_ood_detector():
    def multiply_by_random_matrix(x: torch.Tensor) -> torch.Tensor:
        return x @ torch.randn(2, 5)

    transformations = Transformations(
        [
            multiply_by_random_matrix,
            PCA(n_components=2),
            StandardScaler(),
        ]
    )

    id_train = torch.tensor([[1.0, 1.0], [1.1, 1.0], [0.9, 1.2]])
    id_test = torch.tensor([[1.0, 1.1], [1.2, 0.9]])
    ood_test = torch.tensor([[3.0, 3.0], [4.0, 4.0]])

    detector = OODDetector(
        embedding_function=transformations,
        scoring_function=mahalanobis_distance,
        id_train_data=id_train,
    )

    # Test score
    scores = detector.score(id_test)
    print("Scores for ID test data:", scores)
    assert scores.shape[0] == id_test.shape[0]

    # Test evaluate with value threshold
    threshold = 0.5  # very low, will mark ID as OOD
    cm, tpr, fpr = detector.evaluate(
        id_test, ood_test, threshold=threshold, threshold_type="value"
    )
    print("Confusion matrix:\n", cm)
    print("TPR:", tpr.item(), "FPR:", fpr.item())

    # Test evaluate with percentile threshold
    cm, tpr, fpr = detector.evaluate(id_test, ood_test, threshold_type="percentile")
    print("Confusion matrix (percentile):\n", cm)
    print("TPR:", tpr.item(), "FPR:", fpr.item())


if __name__ == "__main__":
    test_ood_detector()
