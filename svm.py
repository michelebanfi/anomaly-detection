from sklearn.svm import OneClassSVM
import numpy as np

def performSVMAnomalyDetection(dataset, nu):
    svm = OneClassSVM(nu = nu)
    labels = svm.fit_predict(dataset)
    print("[SVM] Founded", len(np.where(labels == -1)[0]), "outliers")
    return labels
