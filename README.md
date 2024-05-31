# Anomaly Detection for Hypertirodism

Install the required libraries by running the following command:

```bash
pip install -r requirements.txt
```

To run the code, execute the following command:

```bash
python main.py
```
The structure of the project is as follows:
```
.
├── Data
│   ├── dataset.csv
│   ├── datasetWithOutliers.csv
│   └── outliers.csv
├── Media
│   ├── <output images>
│   └──[...]
├── Methods
│   ├── DBSCAN.py
│   ├── KNN.py
│   ├── PCA.py
│   └── forest.py
├── Utils
│   ├── descriptive.py
│   └── functions.py
├── main.py
├── README.md
└── requirements.txt
```
Where `dataset.csv` is the original dataset, `datasetWithOutliers.csv` is the original dataset with the outlier probability added to the last column, and `outliers.csv` is the dataset where for each observation we have the label assigned by the algorithm plus the probability of being an outlier and the label after a sharp threshold (see report).
> Note: `datasetWithOutliers.csv` was saved with `,` as separator and `.` as decimal separator in order to be read correctly by pandas.