import numpy as np
import torch
import json

def save_ood_detector_params(ood_detector, filepath):
    """
    Save OOD detector parameters to a file for C++ loading.
    
    Args:
        ood_detector: Fitted OODDetector instance
        filepath: Path to save the parameters (without extension)
    """
    params = {}
    
    # Extract transformations
    transformations = ood_detector.embedding_function.transformations
    
    # 1. PCA parameters
    pca = transformations[1]  # PCA is the second transformation
    params['pca_mean'] = pca.mean_.tolist()  # (D,)
    params['pca_components'] = pca.components_.tolist()  # (n_components, D)
    params['n_pca_components'] = pca.n_components
    
    # 2. StandardScaler parameters
    scaler = transformations[2]  # StandardScaler is the third transformation
    params['scaler_mean'] = scaler.mean_.tolist()  # (n_components,)
    params['scaler_std'] = scaler.scale_.tolist()  # (n_components,)
    
    # 3. Mahalanobis distance parameters
    # embedded_id_train_data has shape (N, n_components)
    embedded_data = ood_detector.embedded_id_train_data
    if isinstance(embedded_data, torch.Tensor):
        embedded_data = embedded_data.cpu().numpy()
    
    # Calculate mean and covariance of embedded ID training data
    mahalanobis_mean = np.mean(embedded_data, axis=0)  # (n_components,)
    centered_data = embedded_data - mahalanobis_mean
    cov = np.cov(centered_data, rowvar=False)  # (n_components, n_components)
    
    # Add regularization to avoid singular matrix
    cov += np.eye(cov.shape[0]) * 1e-6
    
    # Compute inverse covariance
    cov_inv = np.linalg.inv(cov)  # (n_components, n_components)
    
    params['mahalanobis_mean'] = mahalanobis_mean.tolist()
    params['mahalanobis_cov_inv'] = cov_inv.tolist()
    
    # Save as JSON
    with open(f"{filepath}.json", 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"OOD detector parameters saved to {filepath}.json")
    print(f"PCA components: {params['n_pca_components']}")
    print(f"Input dimension: {len(params['pca_mean'])}")