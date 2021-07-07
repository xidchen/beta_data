import collections
import numpy as np
import pandas as pd

import clustering_data
import clustering_utils

np.set_printoptions(precision=2)
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_rows = 12
pd.set_option('max_columns', 12)
pd.set_option('display.max_colwidth', 20)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.unicode.ambiguous_as_wide', True)


# Load and clean data

print('Load and clean data')

beta_data = clustering_data.load_beta_user_profile_data()
original_cols = beta_data.columns.values
print(f'Count of original rows: '
      f'{len(beta_data.index)}')  # 11982911, 15102070
print()


# Preprocess data

print('Preprocess data')

beta_data = beta_data[beta_data['sex'] != 3]
print(f'Count of rows with sex being 0, 1 or 2: '
      f'{len(beta_data.index)}')  # 11982906, 15102065
beta_data = beta_data.dropna(how='all', subset=original_cols[1:])
print(f'Count of rows with 0 NaN columns excluding sex: '
      f'{len(beta_data.index)}')  # 10396709, 13515868
print()

for col in original_cols[1:]:
    print(f'Count of rows with NaN {col}: {beta_data[col].isna().sum()}')
    # location: 156068, 3035547
    # prod2: 9713439, 11076505
    # prod1: 9713439, 11076505
    # asset: 9713439, 11076505
    # topic: 9347087, 9707116
for i in range(5):
    print(f'Count of rows with {i} NaN columns: '
          f'{len(beta_data[beta_data.isna().sum(axis=1) == i].index)}')
    # 0: 542027, 558915
    # 1: 133334, 1725851
    # 2: 7909, 154597
    # 3: 359436, 368887
    # 4: 9354003, 10707618
for i in range(5):
    df = beta_data[beta_data.isna().sum(axis=1) == i]
    print(f'Count of rows with {i} NaN columns excluding location: ', end='')
    print(len(df[df['location'].notna()].index))
    # 0: 542027, 558915
    # 1: 46620, 47992
    # 2: 0, 0
    # 3: 359436, 368887
    # 4: 9292558, 9504527
for col in original_cols:
    print(f'Count of unique {col}: {len(set(beta_data[col]))}')
print()
# Insight: sex, location are must have


# Process location data

loc_count = pd.DataFrame([[k, v] for (k, v) in collections.Counter(
    beta_data['location']).most_common()], columns=['location', 'count'])
loc_count['cum_count'] = loc_count['count'].cumsum()

print('Merge with geo location data')
geo_data = clustering_data.load_geo_location_data()
beta_data = beta_data.merge(geo_data, how='left')
num_quantile = 20
for col in ['lng', 'lat']:
    beta_data[col] = clustering_utils.create_quantiles(
        beta_data[col], num_quantile)
    beta_data[col] = clustering_utils.min_max_scaler(beta_data[col])
    beta_data[col] = beta_data[col].astype('float32')
    print(f'Datatype of {col}: {beta_data[col].dtypes}')
for col in ['lng']:
    df = beta_data[beta_data[col].isna()]
    print(f'Count of rows with NaN lng/lat: '
          f'{len(df.index)}')  # 882012, 3792693
    df = df.drop(['lng', 'lat'], axis=1)
for i in range(5):
    print(f'Count of rows with {i} NaN columns excluding NaN lng/lat: '
          f'{len(df[df.isna().sum(axis=1) == i].index)}')
    # 0: 34196, 36349
    # 1: 89955, 1681290
    # 2: 7909, 154597
    # 3: 24703, 25851
    # 4: 725249, 1894606
beta_data = beta_data.dropna(subset=['lng', 'lat'])
beta_data = beta_data.reset_index(drop=True)
print(f'Count of rows with lng and lat: '
      f'{len(beta_data.index)}')  # 9514697, 9723175
print()


# Process sex and preference data

print('Process sex and preference data')
beta_data['sex'] = beta_data['sex'].astype('int').astype('category')
categorical_cols = np.delete(original_cols, obj=1)
for col in categorical_cols:
    beta_data[col + '2'] = beta_data[col]
encoded_cols = [col + '2' for col in categorical_cols]
beta_data = pd.get_dummies(
    beta_data, prefix=categorical_cols, dummy_na=True, columns=encoded_cols)


# Backup original and process transformed data

print('Backup original and process transformed data')
beta_data_backup = beta_data[original_cols]
beta_data = beta_data.drop(original_cols, axis=1).astype('float32')
print(f'Shape of transformed data: '
      f'{beta_data.shape}')  # (9514697, 313), (9723175, 314)
print()


# Calculate manual similarity

print('Calculate manual similarity')
user0, users = 0, [1, 19]
print('Similarity between user ' + str(user0) + ' and ...')
for i in range(users[0], users[1] + 1):
    print(' ' * 19 + 'user ' + str(i) + ': '
          + str(clustering_utils.get_similarity(beta_data.loc[user0],
                                                beta_data.loc[i])))
print()


# Cluster dataset

print('Cluster dataset')
k = 10
print(f'k: {k}')
beta_feature_cols = beta_data.columns.values
beta_data['centroid'] = k
beta_data['pt2centroid'] = 2
[beta_data, beta_centroids] = clustering_utils.kmeans(
    beta_data, k, beta_feature_cols, 1)
print()
print('Data for the first few users, '
      'with "centroid" and "pt2centroid" on the extreme right:')
print(beta_data)
print()

cluster = 0
print(f'Cluster {cluster}:')
print(beta_data_backup.loc[beta_data['centroid'] == cluster, :])
print()


# Quality metrics for clusters

print('Quality metrics for clusters')
clustering_utils.cluster_quality_metrics(beta_data)
kmin, kmax, kstep = 5, 49, 1
clustering_utils.loss_vs_clusters(
    kmin, kmax, kstep, beta_data, beta_feature_cols)
