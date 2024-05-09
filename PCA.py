import numpy as np
from gower import gower_matrix
from sklearn.decomposition import PCA
from kneed import KneeLocator


def performPCAAnomalyDetection(dataset, NCOMPONENTS):
    # Compute Gower distance matrix for the entire dataset
    gower_dist_matrix = gower_matrix(dataset)

    # Perform PCA
    pca = PCA(n_components=6)
    pca_results = pca.fit_transform(gower_dist_matrix)

    # print PCA explained variance ratio
    print('[PCA] Explained variance ratio: ', pca.explained_variance_ratio_)

    # Compute the reconstruction error for every data point
    X_reconstructed = pca.inverse_transform(pca_results)
    RE = np.linalg.norm(gower_dist_matrix - X_reconstructed, axis=1)

    # Identify outliers using RE
    i = np.arange(len(RE))
    knee = KneeLocator(i, np.sort(RE), S=1, curve='convex', direction='increasing',
                       interp_method='polynomial')

    knee_x = knee.knee
    knee_y = knee.knee_y

    threshold = knee_y
    outliers = np.where(RE > threshold)
    print('[PCA] found %d' % len(outliers[0]))
    pcaLabels = np.zeros(len(dataset))
    pcaLabels[outliers] = -1

    from scipy.stats import chi2
    alpha = 0.99
    chi2_th = chi2.ppf(alpha, NCOMPONENTS)
    print('Chi2 threshold: %.2f' % chi2_th)

    # Identify the outliers
    pcaLabelsChi = np.zeros(len(dataset))
    lambdas = np.sqrt(pca.explained_variance_)
    for i in range(len(dataset)):
        if np.sum(pca_results[i, :] ** 2 / lambdas) > chi2_th:
            pcaLabelsChi[i] = -1

    print('Number of outliers (chi2): %d' % len(np.where(pcaLabelsChi == -1)[0]))

    return pcaLabelsChi

