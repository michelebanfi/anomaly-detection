import numpy as np
from pyod.models.kde import KDE
from sklearn.preprocessing import OneHotEncoder


def performKDEAnomalyDetection(dataset, nu):
    od = KDE(contamination=nu)
    od.fit_predict(dataset)
    labels = od.labels_
    labels = np.where(labels == 1, -1, 0)
    print('[KDE] founded', len(np.where(labels == -1)[0]), "outliers")
    return labels

def performKDEAnomalyDetectionOneHotEncoder(dataset, nu):

    #categorical columns
    categorical_columns = list(range(1, 16))

    #continuos columns
    continuous_columns = [0] + list(range(16, 21))

    encoder = OneHotEncoder()
    encoded_categorical = encoder.fit_transform(dataset.iloc[:, categorical_columns])
    processed_dataset = np.concatenate((encoded_categorical.toarray(), dataset.iloc[:, continuous_columns].values), axis=1)

    od = KDE(contamination=nu)
    od.fit(processed_dataset)
    labels = od.predict(processed_dataset)
    labels = np.where(labels == 1, -1, 0)

    print('[KDE - ONE HOT ENCODE] founded', len(np.where(labels == -1)[0]), "outliers")

    return labels