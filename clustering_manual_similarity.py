import numpy as np
import os
import pandas as pd
import sys

import clustering_data
import clustering_utils

np.set_printoptions(precision=2)
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_rows = 9
pd.set_option('max_columns', 9)
pd.set_option('display.max_colwidth', 9)
pd.set_option('expand_frame_repr', False)


# Load and clean data

print('Load and clean data')

choc_data = clustering_data.load_choc_data()
original_cols = choc_data.columns.values
print(choc_data)


# Process data

print('Preprocess data')
choc_data['rating_norm'] = (
    choc_data['rating'] -
    choc_data['rating'].mean()) / choc_data['rating'].std()
choc_data['cocoa_percent_norm'] = (
    choc_data['cocoa_percent'] -
    choc_data['cocoa_percent'].mean()) / choc_data['cocoa_percent'].std()

countries_info = pd.read_csv(
    'https://download.mlcc.google.com/mledu-datasets/countries_lat_long.csv',
    sep=',', encoding='latin-1')

choc_data = pd.merge(
    choc_data, countries_info, left_on='maker_location', right_on='name')
choc_data.rename(
    columns={
        'longitude': 'maker_longitude',
        'latitude': 'maker_latitude'
    }, inplace=True)
choc_data.drop(columns=['name', 'country'], inplace=True)

choc_data = pd.merge(
    choc_data, countries_info, left_on='broad_origin', right_on='name')
choc_data.rename(
    columns={
        'longitude': 'origin_longitude',
        'latitude': 'origin_latitude'
    },
    inplace=True)
choc_data.drop(
    columns=['name', 'country'], inplace=True)  # don't need this data

num_quantiles = 20
cols_quantiles = ['maker_latitude', 'maker_longitude',
                  'origin_latitude', 'origin_longitude']

for string in cols_quantiles:
    choc_data[string] = clustering_utils.create_quantiles(
        choc_data[string], num_quantiles)
    choc_data[string] = clustering_utils.min_max_scaler(choc_data[string])

print(choc_data)

choc_data['maker2'] = choc_data['maker']
choc_data['bean_type2'] = choc_data['bean_type']
choc_data = pd.get_dummies(
    choc_data, prefix=['maker', 'bean'], columns=['maker2', 'bean_type2'])

choc_data_backup = choc_data.loc[:, original_cols].copy(deep=True)
choc_data.drop(columns=original_cols, inplace=True)
choc_data = choc_data / 1.0


# Cluster chocolate dataset

print('Cluster chocolate dataset')

# Calculate manual similarity

print('Calculate manual similarity')

choc0 = 0
chocs_to_compare = [1, 9]

print('Similarity between chocolate ' + str(choc0) + ' and ...')

for ii in range(chocs_to_compare[0], chocs_to_compare[1] + 1):
    print(str(ii) + ': ' + str(clustering_utils.get_similarity(
        choc_data.loc[choc0], choc_data.loc[ii])))
print('\nFeature data for chocolate ' + str(choc0))
print(choc_data_backup.loc[choc0:choc0, :])
print('\nFeature data for compared chocolates ' + str(chocs_to_compare))
print(choc_data_backup.loc[chocs_to_compare[0]:chocs_to_compare[1], :])
print()


k = 30
choc_feature_cols = choc_data.columns.values  # save original columns
# initialize every point to an impossible value, the k+1 cluster
choc_data['centroid'] = k
# init the point to centroid distance to an impossible value '2' (>1)
choc_data['pt2centroid'] = 2
[choc_data, choc_centroids] = clustering_utils.kmeans(
    choc_data, k, choc_feature_cols, 1)
print()
print('Data for the first few chocolates, '
      'with "centroid" and "pt2centroid" on the extreme right:')
print(choc_data)
print()

cluster = 0
print(f'Cluster {cluster}:')
print(choc_data_backup.loc[choc_data['centroid'] == cluster, :])
print()


# Quality metrics for clusters

print('Quality metrics for clusters')
clustering_utils.cluster_quality_metrics(choc_data)
kmin, kmax, kstep = 5, 80, 1
clustering_utils.loss_vs_clusters(
    kmin, kmax, kstep, choc_data, choc_feature_cols)
