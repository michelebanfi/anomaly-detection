import pandas as pd

def descrptiveStats(dataset):
    print("[DATASET HEAD]", dataset.head())
    print("[DATASET DESCRIBE]", dataset.describe())

    # if the column name ends with a "=0" print the value counts
    for column in dataset.columns:
        if column.endswith("=0"):
            print("[BINARY VALUE COUNTS]", dataset[column].value_counts())

    # plot the scatter matrix of the dataset
    print("[SCATTER MATRIX]")
    pd.plotting.scatter_matrix(dataset, figsize=(40, 40))