import pandas as pd

from GMM import performGMMAnomalyDetection
from descriptive import descrptiveStats
from forest import performIsolationForestAnomalyDetection
from functions import loadDataset, TSNEPlot
from knee import performNNKNEEAnomalyDetection
from lof import performLOFAnomalyDetection

from svm import performSVMAnomalyDetectionOneHotEncoder, performSVMAnomalyDetectionGower
from dbscan import performDBSCANAnomalyDetection

# DBSCAN with Gower distance
dataset = loadDataset()

# description of the dataset
descrptiveStats(dataset)

nu = 0.05
neighborhood_order = 10

# perform dbscan anomaly detection
dbscanLabels = performDBSCANAnomalyDetection(dataset)

# perform svm anomaly detection
svmLabelsOneHotEncode = performSVMAnomalyDetectionOneHotEncoder(dataset, nu)

# perform svm anomaly detection with gower distance
svmLabelsGower = performSVMAnomalyDetectionGower(dataset, nu)

# IsolationForest
forestLabels = performIsolationForestAnomalyDetection(dataset, nu)

# Local Outlier Factor
lofLabels = performLOFAnomalyDetection(dataset, neighborhood_order, nu)

# GMM Outlier Detection
gmmLabels = performGMMAnomalyDetection(dataset, 11, nu)

# KNEE Outlier Detection
kneeLabels = performNNKNEEAnomalyDetection(dataset, neighborhood_order)

# create a dataframe with the labels
df = pd.DataFrame({
    "DBSCAN": dbscanLabels,
    "SVMOneHotEncode": svmLabelsOneHotEncode,
    "SVMGower": svmLabelsGower,
    "IsolationForest": forestLabels,
    "LOF": lofLabels,
    "GMM": gmmLabels,
    "KNEE": kneeLabels
})

# create a new column where we insert the mean of outliers per row
df["Outliers"] = df.mean(axis=1).round(2).abs()


# plot the t-SNE plot
TSNEPlot(dataset, df["Outliers"])

df.to_csv("outliers.csv", index=False)

# normalize the occurrencies of outliers
normalizedOutliers = df["Outliers"].value_counts(normalize=True)


# plot the normalized histogram of the outliers
ax = normalizedOutliers.plot.hist(bins=20, alpha=0.5, title="Outliers Histogram")
fig = ax.get_figure()
fig.savefig("histogram.png")
