import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
import json

# Try to import R's MVN package via rpy2
try:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    import rpy2.robjects as ro
    numpy2ri.activate()
    HAS_MVN = True
    try:
        mvn = importr('MVN')
    except:
        print("Warning: R package 'MVN' not found. Install with: install.packages('MVN') in R")
        HAS_MVN = False
except ImportError:
    print("Warning: rpy2 not installed. Multivariate normality tests unavailable.")
    print("Install with: pip install rpy2")
    HAS_MVN = False

def analyze_layer_distribution(X, layer_name):
    """Analyze the distribution quality of a layer's representations"""
    metrics = {}
    
    # Subsample for computational efficiency if needed
    n_samples_test = min(5000, X.shape[0])
    n_features_test = min(100, X.shape[1])
    
    if X.shape[0] > n_samples_test or X.shape[1] > n_features_test:
        print(f"  Subsampling to {n_samples_test} samples and {n_features_test} features for testing")
        idx_samples = np.random.choice(X.shape[0], n_samples_test, replace=False)
        idx_features = np.random.choice(X.shape[1], n_features_test, replace=False)
        X_test = X[np.ix_(idx_samples, idx_features)]
    else:
        X_test = X
    
    # 1. Multivariate normality tests (if available)
    if HAS_MVN:
        try:
            # Convert to R matrix
            r_data = ro.r.matrix(X_test, nrow=X_test.shape[0], ncol=X_test.shape[1])
            
            # Mardia's test
            mardia_result = mvn.mardiaTest(r_data)
            metrics['mardia_skew_p'] = float(mardia_result.rx2('p.value.skew')[0])
            metrics['mardia_kurt_p'] = float(mardia_result.rx2('p.value.kurt')[0])
            metrics['mardia_p'] = min(metrics['mardia_skew_p'], metrics['mardia_kurt_p'])
            
            # Henze-Zirkler test
            hz_result = mvn.hzTest(r_data)
            metrics['hz_p'] = float(hz_result.rx2('p.value')[0])
            
            # Royston's test (may fail for high dimensions)
            try:
                royston_result = mvn.roystonTest(r_data)
                metrics['royston_p'] = float(royston_result.rx2('p.value')[0])
            except:
                metrics['royston_p'] = None
                
        except Exception as e:
            print(f"  Warning: MVN tests failed: {e}")
            metrics['mardia_p'] = None
            metrics['hz_p'] = None
            metrics['royston_p'] = None
    else:
        # Fallback: Shapiro-Wilk on subset of dimensions
        n_features_shapiro = min(10, X.shape[1])
        idx_features = np.random.choice(X.shape[1], n_features_shapiro, replace=False)
        
        shapiro_stats = []
        for feat_idx in idx_features:
            stat, p = stats.shapiro(X_test[:, feat_idx])
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
        
        # Use multivariate tests if available, otherwise fall back
        if HAS_MVN and m.get('hz_p') is not None:
            # Henze-Zirkler is often most powerful for n >= 75
            # Mardia is good but can be less powerful
            # Royston is good for smaller samples
            gaussianity_score = 0
            if m.get('hz_p') is not None:
                gaussianity_score += m['hz_p'] * 3.0  # Weight HZ most
            if m.get('mardia_p') is not None:
                gaussianity_score += m['mardia_p'] * 2.0
            if m.get('royston_p') is not None:
                gaussianity_score += m['royston_p'] * 1.0
        else:
            # Fallback to Shapiro-Wilk
            gaussianity_score = m.get('mean_shapiro_p', 0) * 2.0
        
        score = (
            gaussianity_score +
            (1.0 / (m['var_coefficient_of_variation'] + 0.1)) * 1.0 +
            m['eigenvalue_entropy'] * 1.0
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
    with open(os.path.join(save_dir, 'best_layer.txt'), 'w') as f:
        f.write(f"{best_layer}\n")
    
    return best_layer, all_metrics

if __name__ == "__main__":
    representations_dir = "representations"
    best_layer, metrics = plot_layer_comparison(representations_dir)
    print(f"\nBest layer saved to analysis/best_layer.txt")