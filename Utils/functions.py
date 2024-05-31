import gower
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import rand_score

# method to load the dataset
def loadDataset(path):
    """
    This function loads the dataset and returns it
    :param path: the path of the dataset
    :return: dataset
    """

    # load the dataset with the correct separator and decimal
    dataset = pd.read_csv(path, sep=";", decimal=",")

    # remove the first column and the last two columns since the first one is the index and the last two are empty
    dataset = dataset.iloc[:, 1:-2]
    return dataset

def TSNEPlot(dist_matrix, labels, path="Media/tsne.png"):
    """
    This function performs the TSNE algorithm to plot the dataset
    :param dist_matrix: the distance matrix of the dataset
    :param neighborhood_order: the labels in the dataset
    :param path: the path where to save the plot
    :return: None
    """

    # perform the TSNE algorithm
    tsne = TSNE(n_components=2, verbose=0, perplexity=20, n_iter=1000, metric="precomputed", init='random', random_state=42)
    tsne_results = tsne.fit_transform(dist_matrix)

    # create the palette and the title of the plot based on the number of labels (outliers score or sharp labels)
    if len(set(labels)) == 2:
        palette = ["#6E1F81", "#FCA06E"]
        title = "Outliers"
    else:
        palette = "plasma_r"
        title = "Outlier Score"

    # plotting
    plt.figure(figsize=(20, 15))
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], palette=palette, hue=labels, legend='full')
    plt.legend(title=title, loc='upper right', markerscale=5, title_fontsize = 30 ,prop = {'size': 30})
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def randScore(dataframe):
    """
    This function calculates the rand matrix on a given dataframe and plots it
    :param dataset: the dataset containing the labels
    :return: None
    """

    print("Calculating the rand matrix...")

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

    # plotting
    plt.figure(figsize=(20, 15))
    sns.set(font_scale=1.6)
    sns.heatmap(rand_matrix, annot=True, xticklabels=cols, yticklabels=cols, cmap="magma", fmt=".2f")
    plt.tight_layout()
    plt.savefig("Media/rand_matrix.png")
    plt.close()

def plotOutliersFrequency(df):
    """
    This function plots the frequency of the outliers in the dataset
    :param df: dataframe containing the labels
    :return: None
    """

    # remove from the Outliers column the value 0
    df = df[df["Outliers"] != 0]

    # get the levels of the outliers
    levels = set(df['Outliers'])

    fig, axs = plt.subplots(2, figsize=(20, 20))
    sns.set(font_scale=2.2)

    # plot the histogram
    sns.histplot(df['Outliers'], bins=len(levels), ax=axs[0])
    axs[0].set_title('Outliers')
    axs[0].set_xlabel('Outliers')
    axs[0].set_ylabel('Frequency')

    # plot the pie chart
    sizes = [len(df[df['Outliers'] == level]) for level in levels]
    palette = sns.color_palette("magma_r", len(levels) + 2)
    palette = palette[:-1]
    axs[1].pie(sizes, labels=levels, autopct='%1.1f%%', colors=palette)

    sns.set_style("whitegrid", {'axes.grid': False})
    plt.tight_layout()
    plt.savefig("Media/outliersFrequency.png")
    plt.close()

def pandas_to_typst(df):
    """
    Custom function to convert a pandas dataframe to a typst table
    :param dataset: the dataframe to plot
    :return: None
    """

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

