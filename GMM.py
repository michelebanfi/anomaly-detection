import numpy as np
from sklearn.mixture import GaussianMixture

def performGMMAnomalyDetection(dataset, components, nu):
    gmm = GaussianMixture(n_components=11, covariance_type='full', random_state=0)
    gmm.fit_predict(dataset)
    # get the means and the covariances
    means = gmm.means_
    covariances = gmm.covariances_
    # get the number of samples
    N = len(dataset)
    # get the number of components
    K = len(means)
    # get the number of features
    M = len(dataset.columns)
    # get the weights
    weights = gmm.weights_
    # get the mahalanobis distance
    gower_distance = np.zeros((N, K))
    for i in range(K):
        cov = covariances[i]
        mean = means[i]
        for j in range(N):
            # calculate the gower distance for each sample
            gower_distance[j, i] = np.sqrt(
                np.dot(np.dot((dataset.iloc[j] - mean).T, np.linalg.inv(cov)), (dataset.iloc[j] - mean)))

    # get the minimum mahalanobis distance
    minMahalanobis = np.min(gower_distance, axis=1)
    # get the threshold
    threshold = np.percentile(minMahalanobis, 100 - nu*100)
    # get the outliers
    outliers = np.where(minMahalanobis > threshold)


    print("[GMM] Founded", len(outliers[0]), "outliers")
    return outliers