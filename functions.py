import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

PAL = ['green', 'blue', 'yellow', 'orange', 'purple', 'magenta', 'cyan', 'brown', 'black', 'red']

def loadDataset():
    # Load dataset
    dataset = pd.read_csv('dataset.csv', sep=";", decimal=",")
    dataset = dataset.iloc[:, 1:-2]
    return dataset

def TSNEPlot(dist_matrix, path, labels):
    tsne = TSNE(n_components=2, verbose=0, perplexity=20, n_iter=300, metric="precomputed", init='random')
    tsne_results = tsne.fit_transform(dist_matrix)

    colors = ['anomaly' if label == -1 else 'normal' for label in labels]
    localPalette = ['green', 'red']
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=colors, palette=localPalette)
    plt.savefig(path)

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
    plt.savefig('silhouette_scores.png')
    return best_eps, best_min_samples, best_score