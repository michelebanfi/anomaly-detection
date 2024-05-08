import gower
import numpy as np
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors as knn

def performNNKNEEAnomalyDetection(dataset, neighborhood_order=10):
    dist_matrix = gower.gower_matrix(dataset)
    # Apply the algorithm
    neighborhood_set = knn(n_neighbors=neighborhood_order).fit(dist_matrix)
    distances, indices = neighborhood_set.kneighbors(dist_matrix)

    i = np.arange(len(distances))
    knee = KneeLocator(i, np.sort(distances[:, -1]), S=1, curve='convex', direction='increasing',
                       interp_method='polynomial')

    knee_x = knee.knee  # x coordinate of the knee point
    knee_y = knee.knee_y  # y coordinate of the knee point

    print('[KNEE] The estimated best eps value is = %.2f' % knee_y, knee_x)

    outliers = np.where(distances[:, neighborhood_order - 1] > 1.5)[0]
    print('[KNEE] Number of outliers: %d' % len(outliers))

    # Assign labels
    kneeLabels = np.zeros(len(dataset))
    kneeLabels[outliers] = -1

    return kneeLabels