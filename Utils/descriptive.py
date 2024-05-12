import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def descriptiveStats(dataset):
    print("[DATASET HEAD]", dataset.head())
    print("[DATASET DESCRIBE]", dataset.describe())

    # plot a histogram for each binary column on a big figure
    plt.figure(figsize=(25, 25))
    for column in dataset.columns:
        if column.endswith("=0"):
            # plot the histogram of the column
            plt.subplot(5, 5, list(dataset.columns).index(column))
            sns.histplot(dataset[column], color='blue', bins=2)
            plt.title(column)
    plt.savefig("Media/binaryDistribution.png")
    plt.close()

    # plot a boxplot for each continuous column on a big figure having 3 rows and 3 columns
    plt.figure(figsize=(15, 15))
    counter = 1
    for column in dataset.columns:
        if not column.endswith("=0"):
            # plot the boxplot of the column
            plt.subplot(3, 3, counter)
            sns.boxplot(y=dataset[column], color='blue')
            plt.title(column)
            counter += 1
    plt.savefig("Media/continuousDistribution.png")
    plt.close()

    # plot the scatter matrix of the dataset
    # print("[SCATTER MATRIX]")
    # pd.plotting.scatter_matrix(dataset, figsize=(40, 40))