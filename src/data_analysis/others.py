# # 3. Mean magnitude
#     metrics["mean_magnitude"] = np.mean(np.linalg.norm(X, axis=1))

#     # 4. Intrinsic dimensionality via PCA
#     n_pca = min(50, X.shape[1])
#     pca = PCA(n_components=n_pca)
#     pca.fit(X)

#     cumsum = np.cumsum(pca.explained_variance_ratio_)
#     metrics["effective_dim_95"] = int(np.argmax(cumsum >= 0.95) + 1)

#     # 5. Eigenvalue entropy
#     eigenvals = pca.explained_variance_ratio_[:20]
#     eigenvals = eigenvals / eigenvals.sum()
#     metrics["eigenvalue_entropy"] = stats.entropy(eigenvals)
