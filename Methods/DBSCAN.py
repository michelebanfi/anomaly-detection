import numpy as np
from sklearn.cluster import DBSCAN
import gower

def performDBSCANAnomalyDetection(dataset, neighborhood_order):
    """
    This function performs the DBSCAN algorithm to detect anomalies in the dataset
    :param dataset: the dataset from which perform the anomaly detection
    :param neighborhood_order: the neighborhood order for the DBSCAN algorithm
    :return: observation labels, where -1 is an outlier and 0 is not an outlier
    """

    dist_matrix = gower.gower_matrix(dataset)
    db = DBSCAN(eps=0.04, min_samples=neighborhood_order, metric="precomputed").fit(dist_matrix)
    labels = db.labels_
    print("[DBSCAN] Number of clusters: ", len(set(labels)) - (1 if -1 in labels else 0))
    print("[DBSCAN] Founded", len(np.where(labels == -1)[0]), "outliers")
    labels = np.where(labels == -1, -1, 0)
    return labels