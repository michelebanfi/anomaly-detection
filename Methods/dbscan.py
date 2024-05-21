import numpy as np
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors as knn
import gower

def performDBSCANAnomalyDetection(dataset, neighborhood_order):
    dist_matrix = gower.gower_matrix(dataset)
    #0.04
    db = DBSCAN(eps=0.2380, min_samples=neighborhood_order, metric="precomputed").fit(dist_matrix)
    labels = db.labels_
    print("[DBSCAN] Number of clusters: ", len(set(labels)) - (1 if -1 in labels else 0))
    print("[DBSCAN] Founded", len(np.where(labels == -1)[0]), "outliers")
    labels = np.where(labels == -1, -1, 0)
    return labels