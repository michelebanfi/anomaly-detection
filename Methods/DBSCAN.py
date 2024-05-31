import numpy as np
from sklearn.cluster import DBSCAN

def performDBSCANAnomalyDetection(dist_matrix, neighborhood_order):
    """
    This function performs the DBSCAN algorithm to detect anomalies in the dataset
    :param dist_matrix: the distance matrix of the dataset
    :param neighborhood_order: the neighborhood order for the DBSCAN algorithm
    :return: observation labels, where -1 is an outlier and 0 is not an outlier
    """

    # perform the DBSCAN algorithm
    db = DBSCAN(eps=0.04, min_samples=neighborhood_order, metric="precomputed").fit(dist_matrix)
    labels = db.labels_
    print("[DBSCAN] Number of clusters: ", len(set(labels)) - (1 if -1 in labels else 0))
    print("[DBSCAN] Founded", len(np.where(labels == -1)[0]), "outliers")

    # assign only two labels: 0 for not outliers and -1 for outliers
    labels = np.where(labels == -1, -1, 0)
    return labels