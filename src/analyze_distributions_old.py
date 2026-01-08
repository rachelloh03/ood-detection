"""When it's fully migrated to analyze_distribution, delete this file."""

import os
from constants import SCRATCH_FILEPATH
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
import json

# Try to import R's MVN package via rpy2
try:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2.robjects as ro

    # Use local converter context instead of deprecated activate()
    HAS_MVN = True
    try:
        mvn = importr("MVN")
    except:
        print(
            "Warning: R package 'MVN' not found. Install with: install.packages('MVN') in R"
        )
        HAS_MVN = False
except ImportError:
    print("Warning: rpy2 not installed. Multivariate normality tests unavailable.")
    print("Install with: pip install rpy2")
    HAS_MVN = False


def analyze_layer_distribution(X, layer_name):
    """Analyze the distribution quality of a layer's representations"""
    metrics = {}

    # Subsample for computational efficiency
    n_samples_test = min(5000, X.shape[0])
    n_features_test = min(100, X.shape[1])

    if X.shape[0] > n_samples_test or X.shape[1] > n_features_test:
        print(
            f"  Subsampling to {n_samples_test} samples and {n_features_test} features for testing"
        )
        idx_samples = np.random.choice(X.shape[0], n_samples_test, replace=False)
        idx_features = np.random.choice(X.shape[1], n_features_test, replace=False)
        X_test = X[np.ix_(idx_samples, idx_features)]
    else:
        X_test = X

    # 1. Multivariate normality tests
    if HAS_MVN:
        with localconverter(ro.default_converter + numpy2ri.converter):
            r_data = X_test  # auto-converts to R matrix

            # --- Mardia ---
            try:
                mardia_result = mvn.mvn(r_data, mvn_test="mardia")
                result_table = r_get(mardia_result, "multivariate_normality")
                p_values = r_get(result_table, "p.value")
                metrics["mardia_skew_p"] = float(p_values[0])
                metrics["mardia_kurt_p"] = (
                    float(p_values[1]) if len(p_values) > 1 else None
                )
                metrics["mardia_p"] = min(
                    metrics["mardia_skew_p"],
                    (
                        metrics["mardia_kurt_p"]
                        if metrics["mardia_kurt_p"] is not None
                        else metrics["mardia_skew_p"]
                    ),
                )
            except Exception as e:
                print(f"    Mardia test failed: {e}")
                metrics["mardia_skew_p"] = None
                metrics["mardia_kurt_p"] = None
                metrics["mardia_p"] = None

            # --- Henze–Zirkler ---
            try:
                hz_result = mvn.mvn(r_data, mvn_test="hz")
                result_table = hz_result.rx2("multivariateNormality")
                metrics["hz_p"] = float(result_table.rx2("p.value")[0])
            except Exception as e:
                print(f"    Henze-Zirkler test failed: {e}")
                metrics["hz_p"] = None

            # --- Royston (often fails in high-d) ---
            try:
                royston_result = mvn.mvn(r_data, mvn_test="royston")
                result_table = royston_result.rx2("multivariateNormality")
                metrics["royston_p"] = float(result_table.rx2("p.value")[0])
            except Exception as e:
                print(f"    Royston test failed: {e}")
                metrics["royston_p"] = None

    else:
        # Fallback: univariate Shapiro–Wilk
        n_features_shapiro = min(10, X_test.shape[1])
        idx_features = np.random.choice(
            X_test.shape[1], n_features_shapiro, replace=False
        )

        shapiro_p = []
        for feat_idx in idx_features:
            _, p = stats.shapiro(X_test[:, feat_idx])
            shapiro_p.append(p)

        metrics["mean_shapiro_p"] = float(np.mean(shapiro_p))

    # 2. Variance balance
    feature_vars = np.var(X, axis=0)
    metrics["var_coefficient_of_variation"] = np.std(feature_vars) / (
        np.mean(feature_vars) + 1e-8
    )

    # 3. Mean magnitude
    metrics["mean_magnitude"] = np.mean(np.linalg.norm(X, axis=1))

    # 4. Intrinsic dimensionality via PCA
    n_pca = min(50, X.shape[1])
    pca = PCA(n_components=n_pca)
    pca.fit(X)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    metrics["effective_dim_95"] = int(np.argmax(cumsum >= 0.95) + 1)

    # 5. Eigenvalue entropy
    eigenvals = pca.explained_variance_ratio_[:20]
    eigenvals = eigenvals / eigenvals.sum()
    metrics["eigenvalue_entropy"] = stats.entropy(eigenvals)

    return metrics


def plot_layer_comparison(representations_dir, save_dir="analysis"):
    """Load all layers and compare their distributions"""
    os.makedirs(save_dir, exist_ok=True)

    layer_files = sorted(
        [f for f in os.listdir(representations_dir) if f.endswith(".npy")]
    )

    all_metrics = {}

    print("Analyzing layer distributions...")
    for layer_file in layer_files:
        layer_name = layer_file.replace(".npy", "")
        print(f"Analyzing {layer_name}...")

        X = np.load(os.path.join(representations_dir, layer_file))
        metrics = analyze_layer_distribution(X, layer_name)
        all_metrics[layer_name] = metrics

        print(f"  {layer_name}:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

    # Save metrics
    with open(os.path.join(save_dir, "layer_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Layer Distribution Analysis", fontsize=16)

    layers = list(all_metrics.keys())
    layer_indices = [int(l.split("_")[1]) for l in layers]

    metric_names = list(all_metrics[layers[0]].keys())

    for idx, metric_name in enumerate(metric_names):
        if idx >= 6:
            break
        ax = axes[idx // 3, idx % 3]

        values = [all_metrics[l][metric_name] for l in layers]
        ax.plot(layer_indices, values, marker="o")
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "layer_comparison.png"), dpi=150)
    print(f"\nPlot saved to {save_dir}/layer_comparison.png")

    # Recommend best layer
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)

    # Score layers (higher is better)
    scores = {}
    for layer in layers:
        m = all_metrics[layer]

        # Use multivariate tests if available, otherwise fall back
        if HAS_MVN and m.get("hz_p") is not None:
            # Henze-Zirkler is often most powerful for n >= 75
            # Mardia is good but can be less powerful
            # Royston is good for smaller samples
            gaussianity_score = 0
            if m.get("hz_p") is not None:
                gaussianity_score += m["hz_p"] * 3.0  # Weight HZ most
            if m.get("mardia_p") is not None:
                gaussianity_score += m["mardia_p"] * 2.0
            if m.get("royston_p") is not None:
                gaussianity_score += m["royston_p"] * 1.0
        else:
            # Fallback to Shapiro-Wilk
            gaussianity_score = m.get("mean_shapiro_p", 0) * 2.0

        score = (
            gaussianity_score
            + (1.0 / (m["var_coefficient_of_variation"] + 0.1)) * 1.0
            + m["eigenvalue_entropy"] * 1.0
        )
        scores[layer] = score

    best_layer = max(scores, key=scores.get)
    print(f"Best layer for OOD detection: {best_layer}")
    print(f"  Score: {scores[best_layer]:.4f}")
    print(f"  Metrics: {all_metrics[best_layer]}")

    if HAS_MVN:
        print("\nMultivariate normality test interpretation:")
        print("  - p > 0.05: Cannot reject normality (good for Mahalanobis)")
        print("  - p < 0.05: Reject normality (may need alternatives)")

    # Save recommendation
    with open(os.path.join(save_dir, "best_layer.txt"), "w") as f:
        f.write(f"{best_layer}\n")

    return best_layer, all_metrics


if __name__ == "__main__":
    best_layer, metrics = plot_layer_comparison(SCRATCH_FILEPATH)
    print("Best layer saved to analysis/best_layer.txt")
