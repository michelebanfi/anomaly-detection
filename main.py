import numpy as np
import pandas as pd
from Methods.PCA import performPCAAnomalyDetection
from Methods.forest import performIsolationForestAnomalyDetection
from Utils.descriptive import descriptiveStats
from Utils.functions import loadDataset, TSNEPlot, randScore, plotOutliersFrequency
from Methods.KNN import performNNKNEEAnomalyDetection
from Methods.DBSCAN import performDBSCANAnomalyDetection

# Load the dataset
dataset = loadDataset()

# description of the dataset
descriptiveStats(dataset)

# Percentage of outliers
nu = 0.05

# neighborhood order of different algorithms
neighborhood_order = 10

# perform dbscan anomaly detection
dbscanLabels = performDBSCANAnomalyDetection(dataset, neighborhood_order)

# IsolationForest
forestLabels = performIsolationForestAnomalyDetection(dataset, nu)

# KNEE Outlier Detection
kneeLabels = performNNKNEEAnomalyDetection(dataset, neighborhood_order)

# PCA Outlier Detection
pcaLabels = performPCAAnomalyDetection(dataset, 10)

# create a dataframe with the labels
df = pd.DataFrame({
    "DBSCAN": dbscanLabels,
    "IsolationForest": forestLabels,
    "KNN": kneeLabels,
    "PCA": pcaLabels,
})

randScore(df)

# create a new column where we insert the mean of outliers per row
df["Outliers"] = df.mean(axis=1).round(2).abs()

# plot the t-SNE plot
TSNEPlot(dataset, df["Outliers"])

# plot the frequency of outliers
plotOutliersFrequency(df)

# print the number of instances that are not considered outliers by any algorithm
print("Found that", len(df[df["Outliers"] == 0]),
      "observations are not considered outliers by any algorithm. Corresponding to the",
      np.round((len(df[df["Outliers"] == 0]) / len(dataset)) * 100, 2), "% of the dataset.")

# take a sharp boundary to consider an observation as an outlier
threshold = 0.6

# create a new column where we flag the observations that have value greater than the threshold
df["SharpOutliers"] = df["Outliers"].apply(lambda x: -1 if x > threshold else 0)

# print the number of instances that are considered outliers by the sharp boundary
print("Found that", len(df[df["SharpOutliers"] == -1]),
      "observations are considered outliers by the sharp boundary. Corresponding to the",
      np.round((len(df[df["SharpOutliers"] == -1]) / len(dataset)) * 100, 2), "% of the dataset.")

df.to_csv("Data/outliers.csv", index=False)

TSNEPlot(dataset, df["SharpOutliers"], "Media/tsneSharp.png")

# add a column to the dataset representing the outlier probability
dataset["OutlierProbability"] = df["Outliers"]

dataset.to_csv("Data/datasetWithOutliers.csv", index=False)






