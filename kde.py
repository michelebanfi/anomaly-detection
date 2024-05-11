import numpy as np
from pyod.models.kde import KDE

def performKDEAnomalyDetection(dataset, nu):
    od = KDE(contamination=nu)
    od.fit_predict(dataset)
    labels = od.labels_
    labels = np.where(labels == 1, -1, 0)
    print('[KDE] founded', len(np.where(labels == -1)[0]), "outliers")
    return labels