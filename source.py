from scipy.spatial import distance
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd


class Centroid:
    def __init__(self, uuid, initial_value):
        self.uuid = uuid
        self.value = initial_value
        self.points = np.array([])

    def update_value(self):
        new_value = np.mean(self.points, axis=0)
        has_changed = not np.array_equal(self.value, new_value)
        self.value = new_value
        return has_changed, self.value

    def point_distance(self, point):
        return distance.euclidean(self.value, point)


def choose_centroids(x, k):
    # Randomly assign k points as centroids
    random_points_indices = np.random.choice(x.shape[0], size=k)
    centroids = [Centroid(i, c) for i, c in enumerate(x[random_points_indices, :])]
    return centroids


def assign_point_to_centroid(point, centroids):
    # Calculate distances
    min_distance = None
    min_centroid = None
    for centroid in centroids:
        if min_distance is not None:
            distance = centroid.point_distance(point)
            if distance < min_distance:
                min_centroid = centroid
                min_distance = distance
        else:
            min_centroid = centroid
            min_distance = centroid.point_distance(point)
    min_centroid.points = np.append(min_centroid.points, point)
    has_changed = min_centroid.update_value()
    return has_changed


def incremental_kmeans(x, k, max_itr=100, random_state=None):
    """
    Inputs:
        x: The data to be clustered (data points)
        k: The number of clusters
        max_itr: The maximum number of iterations
        random_state: Determines the random number generation for centroid initialisation. Use an int to make the randomness deterministic
    Outputs:
        cluster_labels: The cluster membership labels for each element in the data x
        n_iter: The number of iterations run
    """
    # Seed if there is a random state
    if random_state is not None:
        np.random.seed(random_state)

    centroids = choose_centroids(x, k)
    # Initial iteration
    for point in x:
        assign_point_to_centroid(point, centroids)

    # Set to one after initial iteration
    iter_count = 1

    stopping_condition = False

    while not stopping_condition:
        centroids_changed = False
        for point in x:
            centroids_changed = assign_point_to_centroid(point, centroids)
            if not centroids_changed:
                stopping_condition = True

        iter_count += 1
        if iter_count == max_itr:
            stopping_condition = True

    return iter_count


# data = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9],
#     [10, 11, 12],
#     [13, 14, 15]
# ])
dataset = load_iris()

print(incremental_kmeans(dataset.data, 3, random_state=1))
print(dataset)