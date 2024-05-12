import numpy as np
import pandas as pd
from Methods.GMM import performGMMAnomalyDetection
from Methods.PCA import performPCAAnomalyDetection
from Methods.forest import performIsolationForestAnomalyDetection
from Utils.functions import loadDataset, TSNEPlot, randScore, plotOutliersFrequency
from Methods.kde import performKDEAnomalyDetectionOneHotEncoder
from Methods.knee import performNNKNEEAnomalyDetection
from Methods.lof import performLOFAnomalyDetection
from Methods.svm import performSVMAnomalyDetectionOneHotEncoder
from Methods.dbscan import performDBSCANAnomalyDetection

# Load the dataset
dataset = loadDataset()

# description of the dataset
# descrptiveStats(dataset)

# Percentage of outliers
nu = 0.05

# neighborhood order of different algorithms
neighborhood_order = 10

# perform dbscan anomaly detection
dbscanLabels = performDBSCANAnomalyDetection(dataset, neighborhood_order)

# perform svm anomaly detection with one hot encoder
svmLabelsOneHotEncode = performSVMAnomalyDetectionOneHotEncoder(dataset, nu)

# perform svm anomaly detection with gower distance
# svmLabelsGower = performSVMAnomalyDetectionGower(dataset, nu)

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

# KDE Outlier Detection
#kdeLabels = performKDEAnomalyDetection(dataset, nu)

# KDE Outlier Detection with OneHotEncoder
kdeOneHotEncoderLabels = performKDEAnomalyDetectionOneHotEncoder(dataset,nu)

# create a dataframe with the labels
df = pd.DataFrame({
    "DBSCAN": dbscanLabels,
    "SVMOneHotEncode": svmLabelsOneHotEncode,
    #"SVMGower": svmLabelsGower,
    "IsolationForest": forestLabels,
    "LOF": lofLabels,
    "GMM": gmmLabels,
    "KNEE": kneeLabels,
    "PCA": pcaLabels,
    #"KDE": kdeLabels,
    "KDEOneHotEncoder": kdeOneHotEncoderLabels
})

randScore(df)

# create a new column where we insert the mean of outliers per row
df["Outliers"] = df.mean(axis=1).round(2).abs()


# plot the t-SNE plot
TSNEPlot(dataset, df["Outliers"])

df.to_csv("outliers.csv", index=False)

# plot the frequency of outliers
plotOutliersFrequency(df)

# print the number of instances that are not considered outliers by any algorithm
print("Found that", len(df[df["Outliers"] == 0]),
      "observations are not considered outliers by any algorithm. Corresponding to the",
      np.round((len(df[df["Outliers"] == 0]) / len(dataset)) * 100, 2), "% of the dataset.")