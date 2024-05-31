import numpy as np
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors as knn

def performNNKNEEAnomalyDetection(dist_matrix, neighborhood_order):
    """
    This function performs the KNN algorithm to detect anomalies in the dataset
    :param dist_matrix: the distance matrix of the dataset
    :param neighborhood_order: the neighborhood order for the DBSCAN algorithm
    :return: observation labels, where -1 is an outlier and 0 is not an outlier
    """

    # perform the KNN algorithm
    neighborhood_set = knn(n_neighbors=neighborhood_order).fit(dist_matrix)
    distances, indices = neighborhood_set.kneighbors(dist_matrix)

    i = np.arange(len(distances))
    knee = KneeLocator(i, np.sort(distances[:, -1]), S=1, curve='convex', direction='increasing',
                       interp_method='polynomial')

    # y coordinate of the knee point (found visually, see the report)
    knee_y = 1.2

    outliers = np.where(distances[:, neighborhood_order - 1] > knee_y)[0]
    print('[KNN] Founded', len(outliers), "outliers")

    # assign only two labels: 0 for not outliers and -1 for outliers
    kneeLabels = np.zeros(len(dist_matrix))
    kneeLabels[outliers] = -1

    return kneeLabels