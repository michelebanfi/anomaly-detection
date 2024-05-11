from gower import gower_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import OneClassSVM
import numpy as np

def performSVMAnomalyDetectionOneHotEncoder(dataset, nu):

    categorical_columns = list(range(1, 16))  # Categorics attributes
    continuous_columns = [0] + list(range(16, 21))  # Continuos attributes

    encoder = OneHotEncoder()
    encoded_categorical = encoder.fit_transform(dataset.iloc[:, categorical_columns])

    # Attributes concatenation
    processed_dataset = np.concatenate((encoded_categorical.toarray(), dataset.iloc[:, continuous_columns].values),
                                       axis=1)

    svm = OneClassSVM(nu = nu, kernel='rbf')
    labels = svm.fit_predict(processed_dataset)
    print("[SVM - ONE HOT ENCODE] Founded", len(np.where(labels == -1)[0]), "outliers")
    labels = np.where(labels == -1, -1, 0)
    return labels

def performSVMAnomalyDetectionGower(dataset, nu):
    # Compute Gower distance matrix for the entire dataset
    gower_dist_matrix = gower_matrix(dataset)
    # Train One-Class SVM with RBF kernel using Gower distance matrix
    svm = OneClassSVM(nu=nu, kernel='precomputed')
    labels = svm.fit_predict(gower_dist_matrix)

    # Print number of outliers found
    print("[SVM - GOWER DISTANCE] Found", len(np.where(labels == -1)[0]), "outliers")

    labels = np.where(labels == -1, -1, 0)
    return labels