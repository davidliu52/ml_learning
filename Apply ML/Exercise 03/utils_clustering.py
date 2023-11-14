import numpy as np
from matplotlib import pyplot as plt

from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.collections import PatchCollection


def sse(x: np.ndarray, centroids: np.ndarray, labels: np.ndarray, norm: str = 'L2') -> float:
    """ Sum of square errors.

    Defined as sum_K sum_Nk dist(x_i-m_k)^2 with:
        - n-dimensional data space
        - K: number of clusters C
        - Nk: number of points in cluster C_k
        - m_k: centroid of cluster C_k
        - some norm dist

    we assume the data to have a 2D shape:
        - x: [N, n]
        - centroids: [K, n]
        - labels: [N,1]

    """

    # find number of unique cluster labels, i.e. K
    K = np.unique(labels)
    # print(f'unique cluster labels: {K}')

    # dimension of data space
    d = x.shape[1]

    # number of data points
    N = x.shape[0]

    # check correct dimensionalities
    if d is not (centroids.shape[1]):
        raise ValueError('dimensions of data points and centroids mismatch!')

    if N is not (len(labels)):
        raise ValueError(
            'dimensions of data points and cluster labels mismatch!')

    # the distance norm to use
    if norm == 'L1':
        norm = 1
    elif norm == 'L2':
        norm = 2
    else:
        raise ValueError('invalid norm, use L1 or L2!')
    # print(f'using norm {norm}')

    # iterate over all clusters
    sum_of_squares = 0
    for k, cluster in enumerate(K):

        # centroid for current cluster
        m_k = centroids[k]

        # all data points in current cluster
        x_in_cluster = x[(labels == cluster), :]

        # update sum of squares for current cluster and all points within
        sum_of_squares += np.sum(np.linalg.norm(x_in_cluster -
                                 m_k, ord=norm, axis=1)**2)

    return sum_of_squares


def bss(x: np.ndarray, centroids: np.ndarray, norm: str = 'L2') -> float:
    """Compute the between cluster sum of squares.

    Measures the variability in a clustering. Larger values are better.

    x: [N, n] all N data points in n dimensions
    centroids: [K, n] all K centroids in n dimensions

    returns a scalar >0

    """
    # the distance norm to use
    if norm == 'L1':
        norm_ord = 1
    elif norm == 'L2':
        norm_ord = 2
    else:
        raise ValueError('invalid norm, use L1 or L2!')

    # check dimensions
    if x.shape[1] != centroids.shape[1]:
        raise ValueError('dimensions of data points and centroids mismatch!')

    # compute sample mean across all data points
    if norm == 'L1':
        mean_x = np.median(x, axis=0)
    elif norm == 'L2':
        mean_x = np.mean(x, axis=0)

    # sum of squared differences between sample mean and centroids
    bss_value = np.sum(
        (np.linalg.norm(mean_x - centroids, ord=norm_ord, axis=1))**2)

    return bss_value


def pick_random_points(x: np.ndarray, K: int) -> np.ndarray:
    """Pick K random points from a data set.

    Will be used for centroid initialization.

    x: [N, n] N data points in n dimensional space
    K: number of points to randomly pick from x
    """

    rand_idx = np.random.choice(x.shape[0], K, replace=False)
    points = x[rand_idx, :]

    return points


def place_random_points(x: np.ndarray, K: int) -> np.ndarray:
    """ Place K points randomly within data range.

    Will be used for centroid initialization. We are using a uniform
    distribution here, could be extended to Gaussian easily

    x: [N, n]  N data points in n dimensional data space
    K: number of points to randomly choose from the range of x

    returns:
    points: [K, n] array of points
    """

    n = x.shape[1]

    # get value range for each dimension
    range_min = np.min(x, axis=0)
    range_max = np.max(x, axis=0)

    return np.random.uniform(low=range_min, high=range_max, size=(K, n))


def find_farthest_point(x: np.ndarray, y: np.ndarray, norm: str = 'L2') -> np.ndarray:
    """ Find farthest point from point(s) y to point cloud x

    Will be used for centroid placement

    x: [N, n]  N data points in n dimensional data space
    y: [m, n]  m data point(s) from which to maximize the distance
    norm: str  specifying L1 or L2 distance

    returns:
    the point [1, n] in x that has the maximum distance to point(s) y
    """

    if norm == 'L1':
        norm_ord = 1
    elif norm == 'L2':
        norm_ord = 2
    else:
        raise ValueError('choose L1 or L2 as norm!')

    m = y.shape[0]  # number of points y
    dists = []

    # compute distance for each point y to all points in x
    for i in range(m):
        dists.append(np.linalg.norm((y[i, :]-x), ord=norm_ord, axis=1))

    # sum up distances
    dists_sum = np.sum(np.vstack(dists), axis=0)

    # pick the index with maximum distance
    idx_max = np.argmax(dists_sum)
    farthest_point = x[idx_max, :]

    return farthest_point


def iterative_placement(x: np.ndarray, K: int) -> np.ndarray:
    """Select K initial centroids iteratively

    Ensures maximum distance between all centroids.

    Does not involve any random feature, will always produce the same centroids
    for a given set of points!

    x: [N, n]  N data points in n-dimensional space
    K: number of centroids to select

    returns:
    centroids: [K, n] centroids
    """

    # ensure that K is smaller than the number of data points
    N = x.shape[0]
    if K >= N:
        raise ValueError(
            f'Cannot select {K} centroids for {N} data points. Please reduce K')

    # step 1: select sample mean as first centroid
    sample_mean = np.mean(x, axis=0)
    sample_mean = np.expand_dims(sample_mean, axis=0)  # get dimensions right
    # sample_mean = pick_random_points(x=x, K=1)  # for repetitive clusterings: replace sample mean with random point
    centroids = sample_mean

    # step 2-K: select data point the farthest away from the previous centroids
    for _ in range(K-1):
        new_centroid = find_farthest_point(x=x, y=centroids, norm='L2')
        centroids = np.vstack([centroids, new_centroid])

    return centroids


def plot_clusters(x: np.ndarray, labels: np.ndarray = None,
                  centroids: np.ndarray = None,
                  new_figure: bool = True, show_legend: bool = True):
    """Plot clusters in 2D.

    x: coordinates [n x d] in d-dimensional data space
    labels: (optional) cluster assignment [n x 1] for K clusters
    centroids: (optional), [K x d] centroids of clusters

    """

    leg_entries = []

    if new_figure:
        #fig, ax = plt.subplots()
        # fig.set_dpi=200
        fig = plt.figure(figsize=(3, 3), dpi=200)
    else:
        fig = None

    if (labels is None) and (centroids is None):
        # simple scatter plot, e.g. for first data visualization
        plt.scatter(x[:, 0], x[:, 1], marker='o', color='black', alpha=0.3)
        leg_entries.append('all data points')

    elif (labels is not None) and (centroids is None):
        # scatter plot with special markers for each cluster. e.g. ground turth data sets
        # find number of clusters and cluster labels
        cluster_labels = np.unique(labels)
        for cluster in cluster_labels:  # plot data points for each cluster separately
            plt.scatter(x[labels == cluster, 0],
                        x[labels == cluster, 1], alpha=0.5)
            leg_entries.append('cluster ' + str(int(cluster)))

    elif (labels is None) and (centroids is not None):
        # labels not available but centroids available, e.g. for initial centroid placement
        plt.scatter(x[:, 0], x[:, 1], marker='o', color='black', alpha=0.3)
        leg_entries.append('all data points')

        # plot centroids as provided by user
        for k, centroid in enumerate(centroids):
            plt.plot(centroid[0], centroid[1], marker='*', linestyle='none', markersize=15,
                     markerfacecolor='white', markerfacecoloralt='white')
            leg_entries.append('centroid ' + str(int(k)))

    elif (labels is not None) and (centroids is not None):
        # label available and centroid available, e.g. after one iteration of k-means
        # find number of clusters and cluster labels
        cluster_labels = np.unique(labels)
        for cluster in cluster_labels:  # plot data points for each cluster separately
            plt.scatter(x[labels == cluster, 0],
                        x[labels == cluster, 1], alpha=0.5)
            leg_entries.append('cluster ' + str(int(cluster)))

        # plot centroids as provided by user
        for k, centroid in enumerate(centroids):
            plt.plot(centroid[0], centroid[1], marker='*', linestyle='none', markersize=15,
                     markerfacecolor='white', markerfacecoloralt='white')
            leg_entries.append('centroid ' + str(int(k)))

    # plot midpoints of all data points (always)
    # mid_point = np.mean(x, axis=0)
    # plt.plot(mid_point[0], mid_point[1], marker='*', color='gray', linestyle='None', markersize=10)
    # leg_entries.append('data center')

    if show_legend:
        plt.legend(leg_entries, bbox_to_anchor=(1.01, 1.0), loc='upper left')

    plt.xlabel(r'dimension 1')
    plt.ylabel(r'dimension 2')

    return fig
