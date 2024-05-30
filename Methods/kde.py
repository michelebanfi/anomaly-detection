import numpy as np
from pyod.models.kde import KDE

def performKDEAnomalyDetectionOneHotEncoder(dataset, nu):
    # create a function to pass to the metric of the KDE to compute the Gower distance
    od = KDE(contamination=nu, metric="gower")
    od.fit(dataset)
    labels = od.predict(dataset)
    labels = np.where(labels == 1, -1, 0)

    print('[KDE - ONE HOT ENCODE] founded', len(np.where(labels == -1)[0]), "outliers")

    return labels