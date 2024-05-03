import gower
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

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

# create a dataframe with the labels
df = pd.DataFrame({
    "DBSCAN": dbscanLabels,
    "SVM": svmLabels,
    "IsolationForest": forestLabels
})

# create a new column where we compute the number of algorithms that
# detected the sample as an outlier as a number between 0 and 1
df["Outliers"] = (df["DBSCAN"] == -1).astype(int) + (df["SVM"] == -1).astype(int) + (df["IsolationForest"] == -1).astype(int)
df["Outliers"] = df["Outliers"] / 3

# plot the column
# plt.plot(df["Outliers"])
# plt.savefig("outliers.png")

# plot the t-SNE plot
TSNEPlot(dataset, df["Outliers"])

