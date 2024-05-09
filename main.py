import pandas as pd
from matplotlib import pyplot as plt

from GMM import performGMMAnomalyDetection
from PCA import performPCAAnomalyDetection
from descriptive import descrptiveStats
from forest import performIsolationForestAnomalyDetection
from functions import loadDataset, TSNEPlot, randScore
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

# PCA Outlier Detection
pcaLabels = performPCAAnomalyDetection(dataset, 6)

# create a dataframe with the labels
df = pd.DataFrame({
    "DBSCAN": dbscanLabels,
    "SVMOneHotEncode": svmLabelsOneHotEncode,
    "SVMGower": svmLabelsGower,
    "IsolationForest": forestLabels,
    "LOF": lofLabels,
    "GMM": gmmLabels,
    "KNEE": kneeLabels,
    "PCA": pcaLabels
})

randScore(df)

# create a new column where we insert the mean of outliers per row
df["Outliers"] = df.mean(axis=1).round(2).abs()


# plot the t-SNE plot
TSNEPlot(dataset, df["Outliers"])

df.to_csv("outliers.csv", index=False)

fig, axs = plt.subplots(2, figsize=(20, 20))

# Plot histogram
axs[0].hist(df['Outliers'], bins=7, density=True)
axs[0].set_title('Outliers')
axs[0].set_xlabel('Outliers')
axs[0].set_ylabel('Frequency')

# Plot pie chart
levels = set(df['Outliers'])
sizes = [len(df[df['Outliers'] == level]) for level in levels]
axs[1].pie(sizes, labels=levels, autopct='%1.1f%%')

plt.tight_layout()
plt.savefig("media/outliersFrequency.png")
plt.close()
