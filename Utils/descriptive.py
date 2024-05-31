import gower
import matplotlib.pyplot as plt
import seaborn as sns

def descriptiveStats(dataset):
    """
    This function calculates the descriptive statistics of a given dataset
    :param dataset: the dataset to analyze
    :return: None
    """

    print("[DATASET HEAD]", dataset.head())
    print("[DATASET DESCRIBE]", dataset.describe())

    # plot a histogram for each binary column on a big figure
    plt.figure(figsize=(25, 25))
    for column in dataset.columns:
        if column.endswith("=0"):
            # plot the histogram of the column with the class on the x-axis
            plt.subplot(5, 5, list(dataset.columns).index(column))
            sns.countplot(x=column, data=dataset)
            plt.title(column)

    plt.tight_layout()
    plt.savefig("Media/binaryDistribution.png", bbox_inches='tight')
    plt.close()

    # plot a boxplot for each continuous column on a big figure having 3 rows and 3 columns
    plt.figure(figsize=(15, 15))
    counter = 1
    for column in dataset.columns:
        if not column.endswith("=0"):
            # plot the boxplot of the column
            plt.subplot(3, 3, counter)
            sns.boxplot(y=dataset[column])
            plt.title(column)
            counter += 1
    plt.tight_layout()
    plt.savefig("Media/continuousDistributionBoxPlot.png", bbox_inches='tight')
    plt.close()

    # plot the continuos variables distribution
    plt.figure(figsize=(15, 15))
    counter = 1
    for column in dataset.columns:
        if not column.endswith("=0"):
            # plot the distribution of the column
            plt.subplot(3, 3, counter)
            sns.histplot(dataset[column], kde=True)
            plt.title(column)
            counter += 1
    plt.tight_layout()
    plt.savefig("Media/continuousDistributionGaussian.png", bbox_inches='tight')
    plt.close()

    # plot the distance matrix
    dist_matrix = gower.gower_matrix(dataset)
    plt.figure(figsize=(20, 20))
    sns.heatmap(dist_matrix, cmap='plasma_r')
    plt.tight_layout()
    plt.savefig("Media/distanceMatrix.png")
    plt.close()