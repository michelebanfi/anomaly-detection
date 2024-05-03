from forest import performIsolationForestAnomalyDetection
from functions import loadDataset, TSNEPlot

from svm import performSVMAnomalyDetection
from dbscan import performDBSCANAnomalyDetection

# DBSCAN with Gower distance
dataset = loadDataset()

# perform dbscan anomaly detection
dbscanLabels = performDBSCANAnomalyDetection(dataset)

# perform svm anomaly detection
svmLabels = performSVMAnomalyDetection(dataset, 0.03)

# IsolationForest
forestLabels = performIsolationForestAnomalyDetection(dataset, 0.03)