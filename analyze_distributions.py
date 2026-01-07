import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
import json

def analyze_layer_distribution(X, layer_name):
    """Analyze the distribution quality of a layer's representations"""
    metrics = {}
    
    # 1. Gaussianity test (Shapiro-Wilk on first few dimensions)
    # Test on random subset of features (max 10) and samples (max 5000)
    n_features_test = min(10, X.shape[1])
    n_samples_test = min(5000, X.shape[0])
    
    idx_features = np.random.choice(X.shape[1], n_features_test, replace=False)
    idx_samples = np.random.choice(X.shape[0], n_samples_test, replace=False)
    
    shapiro_stats = []
    for feat_idx in idx_features:
        stat, p = stats.shapiro(X[idx_samples, feat_idx])
        shapiro_stats.append(p)
    
    metrics['mean_shapiro_p'] = np.mean(shapiro_stats)
    
    # 2. Variance across dimensions (want balanced variance)
    feature_vars = np.var(X, axis=0)
    metrics['var_coefficient_of_variation'] = np.std(feature_vars) / (np.mean(feature_vars) + 1e-8)
    
    # 3. Mean magnitude (useful for comparing layers)
    metrics['mean_magnitude'] = np.mean(np.linalg.norm(X, axis=1))
    
    # 4. Intrinsic dimensionality (via PCA)
    pca = PCA(n_components=min(50, X.shape[1]))
    pca.fit(X)
    
    # Calculate effective dimensionality (number of components for 95% variance)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum >= 0.95) + 1
    metrics['effective_dim_95'] = n_components_95
    
    # 5. Entropy of eigenvalue distribution (higher = more uniform)
    eigenvals = pca.explained_variance_ratio_[:20]  # Use top 20
    eigenvals = eigenvals / eigenvals.sum()
    metrics['eigenvalue_entropy'] = stats.entropy(eigenvals)
    
    return metrics

def plot_layer_comparison(representations_dir, save_dir="analysis"):
    """Load all layers and compare their distributions"""
    os.makedirs(save_dir, exist_ok=True)
    
    layer_files = sorted([f for f in os.listdir(representations_dir) if f.endswith('.npy')])
    
    all_metrics = {}
    
    print("Analyzing layer distributions...")
    for layer_file in layer_files:
        layer_name = layer_file.replace('.npy', '')
        print(f"Analyzing {layer_name}...")
        
        X = np.load(os.path.join(representations_dir, layer_file))
        metrics = analyze_layer_distribution(X, layer_name)
        all_metrics[layer_name] = metrics
        
        print(f"  {layer_name}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")
    
    # Save metrics
    with open(os.path.join(save_dir, 'layer_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Layer Distribution Analysis', fontsize=16)
    
    layers = list(all_metrics.keys())
    layer_indices = [int(l.split('_')[1]) for l in layers]
    
    metric_names = list(all_metrics[layers[0]].keys())
    
    for idx, metric_name in enumerate(metric_names):
        if idx >= 6:
            break
        ax = axes[idx // 3, idx % 3]
        
        values = [all_metrics[l][metric_name] for l in layers]
        ax.plot(layer_indices, values, marker='o')
        ax.set_xlabel('Layer')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'layer_comparison.png'), dpi=150)
    print(f"\nPlot saved to {save_dir}/layer_comparison.png")
    
    # Recommend best layer
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    
    # Score layers (higher is better)
    scores = {}
    for layer in layers:
        m = all_metrics[layer]
        # Higher Shapiro p-value = more Gaussian
        # Lower var CV = more balanced features
        # Higher entropy = better spread eigenvalues
        score = (
            m['mean_shapiro_p'] * 2.0 +  # Gaussianity (weighted more)
            (1.0 / (m['var_coefficient_of_variation'] + 0.1)) * 1.0 +
            m['eigenvalue_entropy'] * 1.0
        )
        scores[layer] = score
    
    best_layer = max(scores, key=scores.get)
    print(f"Best layer for OOD detection: {best_layer}")
    print(f"  Score: {scores[best_layer]:.4f}")
    print(f"  Metrics: {all_metrics[best_layer]}")
    
    # Save recommendation
    with open(os.path.join(save_dir, 'best_layer.txt'), 'w') as f:
        f.write(f"{best_layer}\n")
    
    return best_layer, all_metrics

if __name__ == "__main__":
    representations_dir = "representations"
    best_layer, metrics = plot_layer_comparison(representations_dir)
    print(f"\nBest layer saved to analysis/best_layer.txt")