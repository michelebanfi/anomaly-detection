import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import chi2

def performPCAAnomalyDetection(dist_matrix, NCOMPONENTS):
    """
    This function performs the PCA algorithm to detect anomalies in the dataset
    :param dist_matrix: the distance matrix of the dataset
    :param NCOMPONENTS: the number of components to reduce the dataset
    :return: observation labels, where -1 is an outlier and 0 is not an outlier
    """

    # perform the PCA algorithm
    pca = PCA(n_components=NCOMPONENTS)
    pca_results = pca.fit_transform(dist_matrix)

    # print the PCA explained variance ratio summed
    print('[PCA] Explained variance ratio up to the ' + str(NCOMPONENTS) + '-th component: ', np.sum(pca.explained_variance_ratio_))

    alpha = 0.05
    chi2_th = chi2.ppf(1 - alpha, NCOMPONENTS)

    # identify the outliers using the chi2 threshold
    pcaLabelsChi = np.zeros(len(dist_matrix))
    lambdas = np.sqrt(pca.explained_variance_)
    for i in range(len(dist_matrix)):
        if np.sum(pca_results[i, :] ** 2 / lambdas) > chi2_th:
            pcaLabelsChi[i] = -1

    print('[PCA] founded', len(np.where(pcaLabelsChi == -1)[0]), "outliers")

    return pcaLabelsChi