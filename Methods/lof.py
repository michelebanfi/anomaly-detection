import gower
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def performLOFAnomalyDetection(dataset, neighbors, nu):
    dist_matrix = gower.gower_matrix(dataset)
    labels = LocalOutlierFactor(n_neighbors=neighbors, metric='precomputed', contamination=nu).fit_predict(dist_matrix)
    print("[LOF] Founded", len(np.where(labels == -1)[0]), "outliers")
    labels = np.where(labels == -1, -1, 0)
    return labels