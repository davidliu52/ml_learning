"""
Custom implementation of the basic K-means algorithm (Lloyd)

(C) Merten Stender, TU Berlin

"""

from matplotlib import pyplot as plt
import numpy as np
from utils_clustering import pick_random_points
from utils_clustering import plot_clusters


def assign_cluster(x: np.ndarray, centroids: np.ndarray, norm: str = 'L2') -> np.ndarray:
    """Assign a cluster index vector to data points given some centroids.

    x:          [N, n]  all data points in n-dimensional space
    centroids:  [K, n]  K centroid coordinates
    norm:       string for the norm: 'L2' or 'L1'

    returns     [N,1] cluster label assignments {0, .. (K-1)}

    The order of the labels will give the order of the clusters, starting from 0
    """

    # number of clusters
    K = centroids.shape[0]

    dists = []  # storing distances from all points to all centroids
    for k in range(K):  # iterate over all K centroids

        # PROBLEM 2: COMPUTE DISTANCE FROM ALL POINTS x TO CURRENT CENTROID
        # append set of distances to list dists by dists.append()

        # convert list into numpy array for better handling
    dists = np.vstack(dists)  # shape: [K, N]

    # PROBLEM 2: for each point in x, find the clostest centroid
    # Check the Numpy method "argmin" in the documentation
    # Return a numpy array that stores cluster indices, ranging from 0 to (K-1)
    cluster_labels =

    return cluster_labels


def update_centroids(x: np.ndarray, labels: np.ndarray, K: int, norm: str = 'L2') -> np.ndarray:
    """Compute new centroid coordinates based on averaging across cluster members.

    x:      data points [N, n]  all data points in n-dimensional space
    labels: cluster assignment vector [N,1] contains {0, ... K-1}; zero-indexed!
    K:      number of clusters K. required as there can also be empty clusters if centroid far away
    norm:   string for the norm: 'L2' or 'L1': mean or median

    returns
    centroids: [K,n] new centroid coordinates

    """
    centroids_new = []  # list of new cluster centroids

    # loop over clusters and re-compute cluster coordinates by averaging
    for k in range(K):

        # boolean index, true when data point belongs to current cluster k
        in_cluster = labels == k

        # PROBLEM 1: IMPLEMENT THE CENTROID UPDATE FOR L1 AND L2 NORM HERE
        # add an entry to a list: <list>.append(<value>)
        # index a Numpy array <a> by a[idx, :] to select rows <idx> from <a>
        cluster_points = x[in_cluster]

        if norm == 'L2':
            centroid_k = np.mean(cluster_points, axis=0)
        elif norm == 'L1':
            centroid_k = np.median(cluster_points, axis=0)
        else:
            raise ValueError("Invalid norm. Use 'L1' or 'L2'.")
        
        centroids_new.append(centroid_k)
        # PROBLEM 5: EXTEND TO HANDLING EMPTY CLUSTERS.
        # any(<array>) will test for any True value in <array>
        # call relocate_empty_centroid if the cluster is empty
        # consider that already the first cluster can be empty (no existing centroids available)

    # convert list of centroids into [K,n] numpy array
    centroids = np.vstack(centroids_new)

    return centroids


def is_converged(centroids: list, labels: list) -> bool:
    """Determine if K-means is converged.

    centroids: [n_iters, [np.ndarray[K, n]]] for the number of previous iterations n_iters, K clusters in n dimensions
    r: [n_iters, [np.ndarray[N,]]] assignment vectors for previous iterations and N data points. contains values 1, .., K

    returns boolean. True if convergence criterion is matched.
    """

    # PROBLEM 3: ADD COMMENTS TO EACH HASHTAG THAT YOU FIND BELOW

    n_iters = len(centroids)    #
    N = labels[0].shape         #

    iters_min = 5               #
    if n_iters < iters_min:
        converged = False       #
    else:

        prev_labels = labels[0]     #
        state = []

        for it in range(n_iters):
            curr_labels = labels[it]  #
            ratio_changed = (N - np.sum(prev_labels == curr_labels)) / N
            state.append(ratio_changed < 0.01)  #
            prev_labels = curr_labels

        converged = all(state[-iters_min:])  #

    return converged


def relocate_empty_centroid(x: np.ndarray, centroids: np.ndarray = None) -> np.ndarray:
    """ Relocate an empty centroid

    We are placing the centroid as far as possible away from any existing centroid

    x: [N, n] N data points in n-dimensional data space
    centroids: [M, n] M existing centroids

    returns
    centroid_new: [1, n] coordinate of a new centroid placed farthes away from
    existing centroids
    """

    if centroids is not None:
        centroid_new = find_farthest_point(x=x, y=centroids)
    else:  # select the sample mean if no other centroids exist
        centroid_new = np.mean(x, axis=0)

    return centroid_new


def kmeans_clustering(x: np.ndarray, K: int, norm: str = 'L2', init_centroids: np.ndarray = None):
    """Basic K-means algorithm.

    x:      [N, n] N data points in n-dimensional data space
    K:      int, number of clusteres desired to find
    norm:   str, ['L2', 'L1'] distance metric to consider

    returns
    labels      [N,1]  final labels
    centroids   [K, n]  finale centroids
    cost        [n_iters]  cost value for each iteration
    """

    # the distance norm to use
    if (norm != 'L1') and (norm != 'L2'):
        raise ValueError('invalid norm, use L1 or L2!')

    # initialization of centroids
    if init_centroids is None:
        # randomly select K points from the data points
        centroids_0 = pick_random_points(x=x, K=K)
    else:
        centroids_0 = init_centroids

    # initialize the return values (list along the iteration)
    centroids = [centroids_0]
    labels = [np.zeros(x.shape[0])]

    converged = False
    i = 0

    # PROBLEM 4: COMPLETE THE CODE BELOW TO IMPLEMENT K-means

    # Both subexpressions must be true for the compound expression to be considered
    # true. If one subexpression is false, then the compound expression is false
    while (??) and (??):  # make sure to catch some infinite looping!

        print(f'K-means iteration {i}')

        # Phase 1: update cluster assignment

        # Phase 2: update centroid positions

        # Check convergence criterion

        # PROBLEM 6: compute cost values SSE and BSS and return them as additional output

    return labels[-1], centroids[-1]  # cost_vals


if __name__ == "__main__":

    """ let's test the functionalities """

    # data points
    x = np.array([[1, 9], [2, 7], [3, 8],
                  [4, 3], [5, 2], [7, 2], [8, 4], [6, 4],
                  [10, 11], [10, 9], [8, 11], [12, 9]])

    # plot the raw data without any clustering
    plot_clusters(x=x)

    # ground truth labels
    labels_gt = np.array([0, 0, 0,
                          1, 1, 1, 1, 1,
                          2, 2, 2, 2])

    # ground truth centroids
    centroids_gt = np.array([[2, 8], [6, 3], [10, 10]])

    # initial centroids
    centroids_0 = np.array([[4, 4], [5, 5], [6, 6]])

    # update cluster assigment
    labels_1 = assign_cluster(x=x, centroids=centroids_0, norm='L2')

    # update centroid positions
    centroids_1 = update_centroids(x=x, labels=labels_1, K=3, norm='L2')

    print(f'new labels: {labels_1}\n')
    print(f'old centroid positions: \n{centroids_0} \n')
    print(f'new centroid positions: \n{centroids_1}')

    # call the K-means clustering algorithm
    labels, centroids = kmeans_clustering(
        x=x, K=3, norm='L2', init_centroids=centroids_0)

    # plot the result
    plot_clusters(x=x, labels=labels, centroids=centroids)
