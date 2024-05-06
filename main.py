import pandas as pd

from GMM import performGMMAnomalyDetection
from descriptive import descrptiveStats
from forest import performIsolationForestAnomalyDetection
from functions import loadDataset, TSNEPlot
from lof import performLOFAnomalyDetection

from svm import performSVMAnomalyDetection
from dbscan import performDBSCANAnomalyDetection

# DBSCAN with Gower distance
dataset = loadDataset()

# description of the dataset
descrptiveStats(dataset)

nu = 0.03

# perform dbscan anomaly detection
dbscanLabels = performDBSCANAnomalyDetection(dataset)

# perform svm anomaly detection
svmLabels = performSVMAnomalyDetection(dataset, nu)

# IsolationForest
forestLabels = performIsolationForestAnomalyDetection(dataset, nu)

# Local Outlier Factor
lofLabels = performLOFAnomalyDetection(dataset, 10, nu)

# GMM Outlier Detection
gmmLabels = performGMMAnomalyDetection(dataset, 11, nu)

# create a dataframe with the labels
df = pd.DataFrame({
    "DBSCAN": dbscanLabels,
    "SVM": svmLabels,
    "IsolationForest": forestLabels,
    "LOF": lofLabels,
    "GMM": gmmLabels
})

# create a new column where we compute the number of algorithms that
# detected the sample as an outlier as a number between 0 and 1
df["Outliers"] = ((df["DBSCAN"] == -1).astype(int) + (df["SVM"] == -1).astype(int) +
                  (df["IsolationForest"] == -1).astype(int) + (df["LOF"] == -1).astype(int)) + (df["GMM"] == -1).astype(int)
df["Outliers"] = df["Outliers"] / 5

# plot the t-SNE plot
TSNEPlot(dataset, df["Outliers"])

