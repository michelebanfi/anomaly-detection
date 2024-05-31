import gower
import numpy as np
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors as knn

def performNNKNEEAnomalyDetection(dataset, neighborhood_order):
    """
    This function performs the KNN algorithm to detect anomalies in the dataset
    :param dataset: the dataset from which perform the anomaly detection
    :param neighborhood_order: the neighborhood order for the DBSCAN algorithm
    :return: observation labels, where -1 is an outlier and 0 is not an outlier
    """

    dist_matrix = gower.gower_matrix(dataset)
    # Apply the algorithm
    neighborhood_set = knn(n_neighbors=neighborhood_order).fit(dist_matrix)
    distances, indices = neighborhood_set.kneighbors(dist_matrix)

    i = np.arange(len(distances))
    knee = KneeLocator(i, np.sort(distances[:, -1]), S=1, curve='convex', direction='increasing',
                       interp_method='polynomial')

    # x coordinate of the knee point
    knee_x = knee.knee
    # y coordinate of the knee point
    knee_y = 1.2

    outliers = np.where(distances[:, neighborhood_order - 1] > knee_y)[0]
    print('[KNN] Founded', len(outliers), "outliers")

    # Assign labels
    kneeLabels = np.zeros(len(dataset))
    kneeLabels[outliers] = -1

    return kneeLabels