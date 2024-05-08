import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder


def performGMMAnomalyDetection(dataset, components, nu):
    categorical_columns = list(range(1, 16))  # Colonne dalla 1 alla 15 che sono categoriche
    continuous_columns = [0] + list(range(16, 21))  # Prima colonna e colonne dalla 16 alla 20

    encoder = OneHotEncoder()
    encoded_categorical = encoder.fit_transform(dataset.iloc[:, categorical_columns])

    # Concatenazione delle variabili categoriche e continue
    processed_dataset = np.concatenate((encoded_categorical.toarray(), dataset.iloc[:, continuous_columns].values),
                                       axis=1)

    gmm = GaussianMixture(n_components=components, covariance_type='full', random_state=0)
    gmm.fit_predict(processed_dataset)
    means = gmm.means_
    covariances = gmm.covariances_
    N = len(processed_dataset) # get the number of samples
    K = len(means) # get the number of components

    # get the minkowski distance
    minkowski = np.zeros((N, K))
    for i in range(K):
        cov = covariances[i]
        mean = means[i]
        for j in range(N):
            # calculate the minkowski distance
            minkowski[j, i] = np.sqrt(np.dot(np.dot((processed_dataset[j] - mean).T, np.linalg.inv(cov)),
                                            (processed_dataset[j] - mean)))

    # get the minimum mahalanobis distance
    minMinkowski = np.min(minkowski, axis=1)
    # get the threshold
    threshold = np.percentile(minMinkowski, 100 - nu * 100)
    # get the outliers
    outliers = np.zeros(N)  # create outliers
    outliers[minMinkowski > threshold] = -1  # Set -1 when I have an outlier
    num_outliers = np.count_nonzero(outliers == -1)  # number of outliers
    print("[GMM] Founded", num_outliers, "outliers")
    return outliers
