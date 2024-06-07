import gower
import numpy as np
import pandas as pd
from Methods.PCA import performPCAAnomalyDetection
from Methods.forest import performIsolationForestAnomalyDetection
from Utils.descriptive import descriptiveStats
from Utils.functions import loadDataset, TSNEPlot, randScore, plotOutliersFrequency
from Methods.KNN import performNNKNEEAnomalyDetection
from Methods.DBSCAN import performDBSCANAnomalyDetection

# load the dataset
# dataset = loadDataset(path = "Data/dataset.csv")
dataset = pd.read_csv("Data/data2_preprocessed.csv")

# remove true labels from the dataset and save them in a separate variable
trueLabels = dataset["anomalies"]
dataset = dataset.drop(columns=["anomalies"])

# compute the Gower distance matrix here, since is used by 3 algorithms
dist_matrix = gower.gower_matrix(dataset)

# description of the dataset
# descriptiveStats(dataset)

# percentage of estimated outliers.
nu = 0.05

# neighborhood order of different algorithms
neighborhood_order = 10

# perform dbscan anomaly detection
dbscanLabels = performDBSCANAnomalyDetection(dist_matrix, neighborhood_order)

# perform IsolationForest anomaly detection
forestLabels = performIsolationForestAnomalyDetection(dataset, nu)

# perform KNEE Outlier Detection
kneeLabels = performNNKNEEAnomalyDetection(dist_matrix, neighborhood_order)

# perform PCA Outlier Detection
pcaLabels = performPCAAnomalyDetection(dist_matrix, 10)

# create a dataframe with the labels
df = pd.DataFrame({
    "DBSCAN": dbscanLabels,
    "IsolationForest": forestLabels,
    "KNN": kneeLabels,
    "PCA": pcaLabels,
})

# compute the Rand Score between the algorithms
randScore(df)

# create a new column for the outlier probability
df["Outliers"] = df.mean(axis=1).round(2).abs()

# plot the t-SNE plot
TSNEPlot(dist_matrix, df["Outliers"])

# plot the frequency of outliers
plotOutliersFrequency(df)

# print the number of instances that are not considered outliers by any algorithm
print("Found that", len(df[df["Outliers"] == 0]),
      "observations are not considered outliers by any algorithm. Corresponding to the",
      np.round((len(df[df["Outliers"] == 0]) / len(dataset)) * 100, 2), "% of the dataset.")

# sharp boundary to consider an observation as an outlier. See the report for more details.
threshold = 0.6

# create a new column where we flag the observations that have value greater than the threshold
df["SharpOutliers"] = df["Outliers"].apply(lambda x: -1 if x > threshold else 0)

# print the number of instances that are considered outliers by the sharp boundary
print("Found that", len(df[df["SharpOutliers"] == -1]),
      "observations are considered outliers by the sharp boundary. Corresponding to the",
      np.round((len(df[df["SharpOutliers"] == -1]) / len(dataset)) * 100, 2), "% of the dataset.")

# save the outliers in a csv file
df.to_csv("Data/BENCHMARK_outliers.csv", index=False)

# plot the t-SNE plot with the sharp boundary
TSNEPlot(dist_matrix, df["SharpOutliers"], "Media/tsneSharp.png")

# add a column to the dataset representing the outlier probability
dataset["OutlierProbability"] = df["Outliers"]

# save the original dataset with the outlier probability
dataset.to_csv("Data/BENCHMARK_datasetWithOutliers.csv", index=False)






