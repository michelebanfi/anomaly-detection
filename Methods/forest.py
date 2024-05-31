import numpy as np
from sklearn.ensemble import IsolationForest

def performIsolationForestAnomalyDetection(dataset, nu):
    """
    This function performs the Isolation Forest algorithm to detect anomalies in the dataset.
    :param dataset: the dataset from which perform the anomaly detection
    :param nu: the percentage of estimated outliers.
    :return: observation labels, where -1 is an outlier and 0 is not an outlier
    """

    forest = IsolationForest(contamination = nu)
    labels = forest.fit_predict(dataset)
    print("[ISOLATION FOREST] Founded", len(np.where(labels == -1)[0]), "outliers")
    labels = np.where(labels == -1, -1, 0)
    return labels