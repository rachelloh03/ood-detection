"""
Run OOD detection for all combinations of pooling functions, subsampling transformations, and scoring functions.

Results in
"""

import itertools
import json
import os
from pathlib import Path
from typing import Callable
from tqdm import tqdm
from sklearn.decomposition import PCA
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from constants.data_constants import JORDAN_DATASET_FILEPATH, MAESTRO_DATASET_FILEPATH
from constants.model_constants import DEVICE, JORDAN_MODEL_NAME
from constants.real_time_constants import SLIDING_WINDOW_LEN, STRIDE
from data.jordan_dataset import JordanDataset
from data.maestro_dataset import MaestroDataset
from data.sliding_window import SlidingWindowDataset
from eval.auroc import get_auroc
from eval.graph_viz import get_graph_visualization
from main.ood_detector import OODDetector
from main.scoring_functions import identity, k_nearest_neighbors, mahalanobis_distance
from main.transformation_functions import (
    GaussianMixtureWithScore,
    KMeansDistance,
    extract_layer_transformation,
    Subsample,
)
from main.transformations import Transformations
from extract_layers.pooling_functions import pool_last_k_tokens, pool_mean_std
from utils.data_loading import collate_fn


def run_ood_detection(
    model,
    transformations: Transformations,
    scoring_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    save_path: str,
):
    stats = {}

    ood_detector = OODDetector(
        embedding_function=transformations,
        scoring_function=scoring_fn,
        id_train_data=id_train_dataloader,
    )

    confusion_matrix, true_positive_rate, false_positive_rate = ood_detector.evaluate(
        id_test_dataloader,
        ood_dataloader,
        threshold=0.7,
        threshold_type="percentile",
    )
    stats["confusion_matrix"] = confusion_matrix.tolist()
    stats["true_positive_rate"] = true_positive_rate.item()
    stats["false_positive_rate"] = false_positive_rate.item()

    auroc = get_auroc(
        ood_detector,
        id_test_dataloader,
        ood_dataloader,
    )

    stats["auroc"] = auroc

    get_graph_visualization(
        ood_detector,
        id_test_dataloader,
        ood_dataloader,
        show_plot=False,
        save_path=save_path,
        output_format="image",
        image_format="png",
    )

    return stats


def load_actual_datasets():
    id_train_base_dataset = JordanDataset(
        data_dir=JORDAN_DATASET_FILEPATH, split="train", name="id_train_base_dataset"
    )
    id_test_base_dataset = JordanDataset(
        data_dir=JORDAN_DATASET_FILEPATH,
        split="validation",
        name="id_test_base_dataset",
        num_samples=80,
    )
    ood_base_dataset = MaestroDataset(
        data_dir=MAESTRO_DATASET_FILEPATH,
        split="test",
        name="maestro_test_base_dataset",
        num_samples=80,
    )

    id_train_dataset = SlidingWindowDataset(
        id_train_base_dataset,
        name="id_train_dataset",
        k=SLIDING_WINDOW_LEN,
        stride=STRIDE,
    )
    print("len(id_train_dataset)", len(id_train_dataset))
    id_test_dataset = SlidingWindowDataset(
        id_test_base_dataset,
        name="id_test_dataset",
        k=SLIDING_WINDOW_LEN,
        stride=STRIDE,
        num_samples=240,
    )
    print("len(id_test_dataset)", len(id_test_dataset))
    ood_dataset = SlidingWindowDataset(
        ood_base_dataset,
        name="ood_dataset",
        k=SLIDING_WINDOW_LEN,
        stride=STRIDE,
        num_samples=240,
    )
    print("len(ood_dataset)", len(ood_dataset))

    return id_train_dataset, id_test_dataset, ood_dataset


def load_dummy_datasets():
    id_train_base_dataset = JordanDataset(
        data_dir=JORDAN_DATASET_FILEPATH,
        split="train",
        name="id_train_dummy_dataset",
        num_samples=10,
    )
    id_test_base_dataset = JordanDataset(
        data_dir=JORDAN_DATASET_FILEPATH,
        split="validation",
        name="id_test_dummy_dataset",
        num_samples=10,
    )
    ood_base_dataset = MaestroDataset(
        data_dir=MAESTRO_DATASET_FILEPATH,
        split="test",
        name="maestro_test_dummy_dataset",
        num_samples=10,
    )

    id_train_dataset = SlidingWindowDataset(
        id_train_base_dataset,
        name="id_train_dummy_dataset",
        k=SLIDING_WINDOW_LEN,
        stride=STRIDE,
        num_samples=100,
    )
    id_test_dataset = SlidingWindowDataset(
        id_test_base_dataset,
        name="id_test_dummy_dataset",
        k=SLIDING_WINDOW_LEN,
        stride=STRIDE,
        num_samples=16,
    )
    ood_dataset = SlidingWindowDataset(
        ood_base_dataset,
        name="ood_dummy_dataset",
        k=SLIDING_WINDOW_LEN,
        stride=STRIDE,
        num_samples=16,
    )

    return id_train_dataset, id_test_dataset, ood_dataset


def generate_all_transformations_experiment_1(
    model, pooling_functions, subsampling_transformations
):
    all_transformations = []
    for pooling_function, pooling_function_name in pooling_functions:
        for n_clusters in [25, 10, 5]:
            all_transformations.append(
                (
                    Transformations(
                        [
                            extract_layer_transformation(model, pooling_function, [12]),
                            StandardScaler(),
                            KMeansDistance(
                                n_clusters=n_clusters
                            ),  # (n_samples, n_clusters),
                            lambda x: x.mean(dim=1),  # (n_samples, 1)
                        ]
                    ),
                    f"{pooling_function_name}_KMeansDistance(n_clusters={n_clusters})_withmean",
                )
            )
            all_transformations.append(
                (
                    Transformations(
                        [
                            extract_layer_transformation(model, pooling_function, [12]),
                            StandardScaler(),
                            KMeansDistance(
                                n_clusters=n_clusters
                            ),  # (n_samples, n_clusters),
                        ]
                    ),
                    f"{pooling_function_name}_KMeansDistance(n_clusters={n_clusters})_withoutmean",
                )
            )

    for pooling_function, pooling_function_name in pooling_functions:
        for (
            subsampling_transformation,
            subsampling_transformation_name,
        ) in subsampling_transformations:
            all_transformations.append(
                (
                    Transformations(
                        [
                            extract_layer_transformation(model, pooling_function, [12]),
                            StandardScaler(),
                            subsampling_transformation,
                        ]
                    ),
                    f"{pooling_function_name}_{subsampling_transformation_name}",
                )
            )

    return all_transformations


def generate_all_transformations_experiment_2(
    model,
    components_to_try=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    pca_components_to_try=[50],
    diagonal_cov_to_try=[True, False],
):
    """GMM + MD"""
    all_transformations = []
    for layer in range(17, 18):
        for components in components_to_try:
            for pca_components in pca_components_to_try:
                for diagonal_cov in diagonal_cov_to_try:
                    all_transformations.append(
                        (
                            Transformations(
                                [
                                    extract_layer_transformation(
                                        model, pool_mean_std, [layer]
                                    ),
                                    StandardScaler(),
                                    PCA(n_components=pca_components),
                                    GaussianMixtureWithScore(
                                        n_components=components,
                                        dim=pca_components,
                                        diagonal_cov=diagonal_cov,
                                    ),  # (n_samples,)
                                ]
                            ),
                            f"GMM(layer={layer}, n_components={components}, pca_components={pca_components},"
                            + f"diagonal_cov={diagonal_cov})",
                        )
                    )
    return all_transformations


if __name__ == "__main__":
    print("running")
    id_train_dataset, id_test_dataset, ood_dataset = load_actual_datasets()

    batch_size = 8

    id_train_dataloader = DataLoader(
        id_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    id_test_dataloader = DataLoader(
        id_test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    ood_dataloader = DataLoader(
        ood_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    model = AutoModelForCausalLM.from_pretrained(
        JORDAN_MODEL_NAME,
        dtype=torch.float32,
    ).to(DEVICE)

    pooling_functions = [
        (pool_mean_std, "pool_mean_std"),
        (pool_last_k_tokens(k=10), "pool_last_k_tokens(k=10)"),
        (pool_last_k_tokens(k=5), "pool_last_k_tokens(k=5)"),
        (pool_last_k_tokens(k=1), "pool_last_k_tokens(k=1)"),
    ]
    subsampling_transformations = [
        (Subsample(50), "Subsample(50)"),
        (PCA(n_components=50), "PCA(n_components=50)"),
        (Subsample(20), "Subsample(20)"),
        (PCA(n_components=20), "PCA(n_components=20)"),
        (Subsample(10), "Subsample(10)"),
        (PCA(n_components=10), "PCA(n_components=10)"),
    ]
    scoring_functions = [
        # (k_nearest_neighbors(k=50), "knn(k=50)"),
        # (k_nearest_neighbors(k=20), "knn(k=20)"),
        # (k_nearest_neighbors(k=10), "knn(k=10)"),
        # (k_nearest_neighbors(k=5), "knn(k=5)"),
        # (k_nearest_neighbors(k=3), "knn(k=3)"),
        # (k_nearest_neighbors(k=2), "knn(k=2)"),
        # (k_nearest_neighbors(k=1), "knn(k=1)"),
        # (mahalanobis_distance, "mahalanobis_distance"),
        (identity, "identity"),
    ]
    stats_file = "results/stats.json"
    all_stats = []

    graphs_dir = "./output/graphs_test"
    Path(graphs_dir).mkdir(parents=True, exist_ok=True)

    try:
        with open(stats_file, "r") as f:
            all_stats = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if isinstance(e, json.JSONDecodeError):
            print(
                f"Warning: {stats_file} is corrupted or invalid JSON. Starting with empty stats."
            )
            print(f"Error details: {e}")
        all_stats = []

    all_transformations = generate_all_transformations_experiment_2(model)

    total_iterations = len(all_transformations) * len(scoring_functions)
    for (transformations, transformations_name), (scoring_fn, scoring_fn_name) in tqdm(
        itertools.product(all_transformations, scoring_functions),
        total=total_iterations,
    ):

        save_path = os.path.join(
            graphs_dir,
            f"graph_visualization_{transformations_name}_{scoring_fn_name}.png",
        )

        stats = run_ood_detection(model, transformations, scoring_fn, save_path)
        print(
            f"Transformations: {transformations_name},"
            f"Scoring function: {scoring_fn_name}"
        )
        print(f"Stats: {stats}")

        stats_entry = {
            "transformations": transformations_name,
            "scoring_function": scoring_fn_name,
            **stats,
        }

        all_stats.append(stats_entry)

        with open(stats_file, "w") as f:
            json.dump(all_stats, f, indent=2)
