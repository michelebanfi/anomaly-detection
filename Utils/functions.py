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
    path = "Data/dataset.csv"
    # Load dataset
    dataset = pd.read_csv(path, sep=";", decimal=",")
    dataset = dataset.iloc[:, 1:-2]
    return dataset

def TSNEPlot(dataset, labels, path="Media/tsne.png"):
    dist_matrix = gower.gower_matrix(dataset)
    tsne = TSNE(n_components=2, verbose=0, perplexity=20, n_iter=1000, metric="precomputed", init='random', random_state=42)
    tsne_results = tsne.fit_transform(dist_matrix)

    if len(set(labels)) == 2:
        palette = ["#6E1F81", "#FCA06E"]
        title = "Outliers Score"
    else:
        palette = "plasma_r"
        title = "Outliers"

    # create a big figure with a white background
    plt.figure(figsize=(20, 15))
    #sns.set(font_scale=3)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], palette=palette, hue=labels, legend='full')
    plt.legend(title=title, loc='upper right', prop = {'size': 20})
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

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
    print("Calculating the rand matrix...")
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

    # plot the rand matrix with the labels of the algorithms with a bigger font size
    plt.figure(figsize=(20, 15))
    sns.set(font_scale=1.6)
    sns.heatmap(rand_matrix, annot=True, xticklabels=cols, yticklabels=cols, cmap="magma", fmt=".2f")
    plt.tight_layout()
    plt.savefig("Media/rand_matrix.png")
    plt.close()

def plotOutliersFrequency(df):
    # remove from the Outliers column the value 0
    df = df[df["Outliers"] != 0]

    levels = set(df['Outliers'])

    fig, axs = plt.subplots(2, figsize=(20, 20))
    sns.set(font_scale=2.2)
    # Plot histogram
    sns.histplot(df['Outliers'], bins=len(levels), ax=axs[0])
    axs[0].set_title('Outliers')
    axs[0].set_xlabel('Outliers')
    axs[0].set_ylabel('Frequency')

    # Plot pie chart
    sizes = [len(df[df['Outliers'] == level]) for level in levels]
    palette = sns.color_palette("magma_r", len(levels) + 2)
    palette = palette[:-1]
    axs[1].pie(sizes, labels=levels, autopct='%1.1f%%', colors=palette)

    sns.set_style("whitegrid", {'axes.grid': False})
    plt.tight_layout()
    plt.savefig("Media/outliersFrequency.png")
    plt.close()

def pandas_to_typst(df):
    #USAGE: print(pandas_to_typst(df))

    typst_text = "\n#table(\n"

    # Get number of columns
    num_columns = len(df.columns)
    typst_text += f"  columns: {num_columns},\n"

    # Get column names
    typst_text += "  "
    for col in df.columns:
        typst_text += f"[*{col}*], "
    typst_text = typst_text[:-2] + ",\n"

    # Get values
    for index, row in df.iterrows():
        typst_text += "  "
        for value in row:
            typst_text += f"[{value}], "
        typst_text = typst_text[:-2] + ",\n"

    typst_text += ") \n"

    return typst_text

