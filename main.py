import pandas as pd

from forest import performIsolationForestAnomalyDetection
from functions import loadDataset, TSNEPlot
from lof import performLOFAnomalyDetection

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

# Local Outlier Factor
lofLabels = performLOFAnomalyDetection(dataset, 10)

# create a dataframe with the labels
df = pd.DataFrame({
    "DBSCAN": dbscanLabels,
    "SVM": svmLabels,
    "IsolationForest": forestLabels,
    "LOF": lofLabels
})

# create a new column where we compute the number of algorithms that
# detected the sample as an outlier as a number between 0 and 1
df["Outliers"] = ((df["DBSCAN"] == -1).astype(int) + (df["SVM"] == -1).astype(int) +
                  (df["IsolationForest"] == -1).astype(int) + (df["LOF"] == -1).astype(int))
df["Outliers"] = df["Outliers"] / 4

# plot the column
# plt.plot(df["Outliers"])
# plt.savefig("outliers.png")

# plot the t-SNE plot
TSNEPlot(dataset, df["Outliers"])

