import gower
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, rand_score
from sklearn.cluster import DBSCAN

PAL = ['green', 'blue', 'yellow', 'orange', 'purple', 'magenta', 'cyan', 'brown', 'black', 'red']

def loadDataset():
    # Load dataset
    dataset = pd.read_csv('dataset.csv', sep=";", decimal=",")
    dataset = dataset.iloc[:, 1:-2]
    return dataset

def TSNEPlot(dataset, labels):
    dist_matrix = gower.gower_matrix(dataset)
    tsne = TSNE(n_components=2, verbose=0, perplexity=20, n_iter=1000, metric="precomputed", init='random')
    tsne_results = tsne.fit_transform(dist_matrix)

    # create a big figure
    plt.figure(figsize=(20, 15))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette=PAL, legend='full')
    plt.savefig("media/tsne.png")
    plt.close()
# call the gridSearchDBSCAN method to find the best parameters for the DBSCAN clustering algorithm
# eps = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
# min_samples = [2, 3, 4, 5, 6, 7, 9, 11, 15, 20]
# best_eps, best_min_samples, best_score = gridSearchDBSCAN(dist_matrix, min_samples, eps)
# print(best_eps, best_min_samples, best_score)

# create a method to perform a grid search for DBSCAN with gower distance in order to obtain the best parameters for the
# clustering algorithm. Uisng the silhouette score as the metric to evaluate the performance of the clustering algorithm.
def gridSearchDBSCAN(dist_matrix, min_samples, eps):
    silhouette_scores = []
    best_score = -1
    best_eps = 0
    best_min_samples = 0
    for eps in eps:
        for sample in min_samples:
            db = DBSCAN(eps = eps, min_samples = sample, metric = "precomputed").fit(dist_matrix)
            labels = db.labels_
            silhouette = silhouette_score(dist_matrix, labels, metric='precomputed')
            silhouette_scores.append(silhouette)
            if silhouette > best_score:
                best_score = silhouette
                best_eps = eps
                best_min_samples = sample
    plt.plot(silhouette_scores)
    return best_eps, best_min_samples, best_score



def randScore(dataframe):
    # calculate the rand score of multiple algorithms saved in columns of a pandas dataframe
    # get the number of columns
    cols = dataframe.columns
    n = len(cols)
    # create a matrix to store the rand scores
    rand_matrix = np.zeros((n, n))
    # iterate over the columns
    for i in range(n):
        for j in range(n):
            # get the labels
            labels1 = dataframe[cols[i]]
            labels2 = dataframe[cols[j]]
            # calculate the rand score
            rand_matrix[i, j] = rand_score(labels1, labels2)
            print("Rand score between", cols[i], "and", cols[j], "is", rand_matrix[i, j])


    # plot the rand matrix with the labels of the algorithms
    sns.heatmap(rand_matrix, annot=True, xticklabels=cols, yticklabels=cols)
    plt.savefig("media/rand_matrix.png")
    plt.close()






