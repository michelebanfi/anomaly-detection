import numpy as np
from gower import gower_matrix
from sklearn.decomposition import PCA
from scipy.stats import chi2

def performPCAAnomalyDetection(dataset, NCOMPONENTS):
    # Compute Gower distance matrix for the entire dataset
    gower_dist_matrix = gower_matrix(dataset)

    # Perform PCA
    pca = PCA(n_components=NCOMPONENTS)
    pca_results = pca.fit_transform(gower_dist_matrix)

    # print the PCA explained variance ratio summed
    print('[PCA] Explained variance ratio: ', np.sum(pca.explained_variance_ratio_))

    # Compute the reconstruction error for every data point
    X_reconstructed = pca.inverse_transform(pca_results)
    RE = np.linalg.norm(gower_dist_matrix - X_reconstructed, axis=1)

    alpha = 0.05
    chi2_th = chi2.ppf(1 - alpha, NCOMPONENTS)

    # Identify the outliers
    pcaLabelsChi = np.zeros(len(dataset))
    lambdas = np.sqrt(pca.explained_variance_)
    for i in range(len(dataset)):
        if np.sum(pca_results[i, :] ** 2 / lambdas) > chi2_th:
            pcaLabelsChi[i] = -1

    print('[PCA] founded', len(np.where(pcaLabelsChi == -1)[0]), "outliers")

    return pcaLabelsChi