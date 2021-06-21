import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_quantiles(_df_column, _num_quantiles):
    return pd.qcut(_df_column, _num_quantiles, labels=False, duplicates='drop')


def min_max_scaler(num_arr):
    return (num_arr - np.min(num_arr)) / (np.max(num_arr) - np.min(num_arr))


def get_similarity(obj1, obj2):
    if not len(obj1.index) == len(obj2.index):
        print('Error: Compared objects must have same number of features.')
        sys.exit(1)
    else:
        similarity = np.sum(((obj1 - obj2) ** 2) / 10)
        return round(1 - np.sqrt(similarity), 4)


def df_similarity(df, centroids):
    """Calculate similarities for dataframe input"""
    point_norms = np.square(np.linalg.norm(df, axis=1))
    point_norms = np.reshape(point_norms, [len(df.index), 1])
    centroid_norms = np.square(np.linalg.norm(centroids, axis=1))
    centroid_norms = np.reshape(centroid_norms, (1, len(centroids.index)))
    similarities = point_norms + centroid_norms - 2.0 * np.dot(
        df, np.transpose(centroids))
    # Divide by the number of features which is 10
    # because the one-hot encoding means
    # the 'Maker' and 'Bean' are weighted twice
    similarities = similarities / 10.0
    # Numerical artifacts lead to negligible
    # but negative values that go to NaN on the root
    similarities = similarities.clip(min=0.0)
    # Square root since it's ||a-b||^2
    return np.sqrt(similarities)


def init_centroids(df, _k, feature_cols):
    """Pick '_k' random examples to serve as initial centroids"""
    centroids_key = np.random.randint(0, len(df.index) - 1, _k)
    centroids = df.loc[centroids_key, feature_cols].copy()
    # the indexes get copied over so reset them
    return centroids.reset_index(drop=True)


def pt2centroid(df, centroids, feature_cols):
    """Calculate similariteis between all points and centroids and
    assign points to the closest centroid and save that distance"""
    dist = df_similarity(df.loc[:, feature_cols],
                         centroids.loc[:, feature_cols])
    df.loc[:, 'centroid'] = np.argmin(dist, axis=1)  # closest centroid
    df.loc[:, 'pt2centroid'] = np.min(dist, axis=1)  # minimum distance
    return df


def recompute_centroids(df, centroids, feature_cols):
    """Recompute each centroid as an average of the points assigned to it"""
    for cen in range(len(centroids.index)):
        df_subset = df.loc[df['centroid'] == cen, feature_cols]
        if not df_subset.empty:  # if there are points assigned to the centroid
            centroids.loc[cen] = np.sum(df_subset) / len(df_subset.index)
    return centroids


def kmeans(df, _k, feature_cols, verbose):
    flag_convergence = False
    max_iter = 100
    i = 0  # ensure kmeans doesn't run forever
    centroids = init_centroids(df, _k, feature_cols)
    while not flag_convergence:
        i += 1
        # Save old mapping of points to centroids
        old_mapping = df['centroid'].copy()
        # Perform k-means
        df = pt2centroid(df, centroids, feature_cols)
        centroids = recompute_centroids(df, centroids, feature_cols)
        # Check convergence by comparing [old_mapping, new_mapping]
        new_mapping = df['centroid']
        flag_convergence = all(pd.Series(old_mapping) == pd.Series(new_mapping))
        if verbose == 1:
            print('Total distance: ' +
                  str(round(float(np.sum(df['pt2centroid'])), 2)))
        if i > max_iter:
            print('k-means did not converge! '
                  'Reached maximum iteration limit of '
                  + str(max_iter) + '.')
            sys.exit(1)
    print('k-means converged for ' + str(_k) +
          ' clusters after ' + str(i) + ' iterations!')
    return [df, centroids]


def cluster_cardinality(df):
    _k = int(np.max(df['centroid']) + 1)
    cl_card = np.zeros(_k)
    for kk in range(_k):
        cl_card[kk] = np.sum(df['centroid'] == kk)
    cl_card = cl_card.astype(int)
    print('Cluster cardinality: ' + str(cl_card))
    plt.figure()
    plt.bar(range(_k), cl_card)
    plt.title('Cluster Cardinality')
    plt.xlabel('Cluster Number: ' + str(0) + ' to ' + str(_k - 1))
    plt.ylabel('Points in Cluster')
    plt.show()
    return cl_card


def cluster_magnitude(df):
    _k = int(np.max(df['centroid']) + 1)
    cl_mag = np.zeros(_k)
    for kk in range(_k):
        idx = np.where(df['centroid'] == kk)[0]
        cl_mag[kk] = np.sum(df.loc[idx, 'pt2centroid'])
    print('Cluster magnitude: ', cl_mag)
    plt.figure()
    plt.bar(range(_k), cl_mag)
    plt.title('Cluster Magnitude')
    plt.xlabel('Cluster Number: ' + str(0) + ' to ' + str(_k - 1))
    plt.ylabel('Total Point-to-Centroid Distance')
    plt.show()
    return cl_mag


def plot_card_vs_mag(cl_card, cl_mag):
    plt.figure()
    plt.scatter(cl_card, cl_mag)
    for i in range(len(cl_card)):
        plt.annotate(str(i), (cl_card[i], cl_mag[i]))
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.title('Magnitude vs Cardinality')
    plt.ylabel('Magnitude')
    plt.xlabel('Cardinality')
    plt.show()


def cluster_quality_metrics(df):
    cl_card = cluster_cardinality(df)
    cl_mag = cluster_magnitude(df)
    plot_card_vs_mag(cl_card, cl_mag)


def loss_vs_clusters(_kmin, _kmax, _kstep, _df, _feature_cols):
    k_range = range(_kmin, _kmax + 1, _kstep)
    loss = np.zeros(len(k_range))
    loss_ctr = 0
    for k in k_range:
        print(f'Number of clusters: {k}')
        [_df, _] = kmeans(_df, k, _feature_cols, 0)
        cluster_quality_metrics(_df)
        loss[loss_ctr] = np.sum(_df['pt2centroid'])
        loss_ctr += 1
    plt.scatter(k_range, loss)
    plt.title('Loss vs Clusters Used')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Total Point-to-Centroid Distance')
    plt.show()
