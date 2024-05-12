import numpy as np
from sklearn.ensemble import IsolationForest

def performIsolationForestAnomalyDetection(dataset, nu):
    forest = IsolationForest(contamination = nu)
    labels = forest.fit_predict(dataset)
    print("[ISOLATION FOREST] Founded", len(np.where(labels == -1)[0]), "outliers")
    labels = np.where(labels == -1, -1, 0)
    return labels