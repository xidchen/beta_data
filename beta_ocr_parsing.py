import numpy as np
import pandas as pd
import sklearn.cluster as sc

import beta_utils


DIST_THRESH = 25.0
MIN_SIZE = 2


def read_ocr_result(file_path: str) -> {}:
    """Read JSON string of OCR result
    :param file_path: file path of JSON string
    :return: dict of JSON
    """
    with open(file_path, encoding='utf-8') as f:
        json_str = f.readline()
    return eval(json_str)


def parse_ocr_result(json_dict: {}) -> [pd.DataFrame]:
    """Parse OCR result into a list of DataFrame
    :param json_dict: JSON dict of OCR result
    :return: list of table DataFrame
    """
    lodf = []
    for table_result in json_dict['tables_result']:
        df = parse_table_result(table_result)
        lodf.append(df)
    return lodf


def parse_table_result(t_result: {}) -> pd.DataFrame:
    """Parse OCR result of one table into a DataFrame
    :param t_result: JSON dict of table's OCR result
    :return: DataFrame of the table
    """
    t_body = t_result['body']
    coords = [(c['cell_location'][0]['x'],
               c['cell_location'][0]['y'],
               c['cell_location'][2]['x'],
               c['cell_location'][2]['y']) for c in t_body]
    texts = [c['words'] for c in t_body]
    x_coords = [(c[0], 0) for c in coords]
    clustering = sc.AgglomerativeClustering(n_clusters=None,
                                            affinity='manhattan',
                                            linkage='complete',
                                            distance_threshold=DIST_THRESH)
    clustering.fit(x_coords)
    sorted_clusters = []
    for label in np.unique(clustering.labels_):
        indices = np.where(clustering.labels_ == label)[0]
        if len(indices) > MIN_SIZE:
            avg = np.average([x_coords[i][0] for i in indices])
            sorted_clusters.append((label, avg))
    sorted_clusters.sort(key=lambda x: x[1])
    df = pd.DataFrame()
    for label, _ in sorted_clusters:
        indices = np.where(clustering.labels_ == label)[0]
        y_coords = [coords[i][1] for i in indices]
        sorted_indices = indices[np.argsort(y_coords)]
        cols = [texts[i].strip() for i in sorted_indices]
        current_df = pd.DataFrame({cols[0]: cols[1:]})
        df = pd.concat([df, current_df], axis=1)
    df = drop_useless_char_in_df(df)
    df = df.drop('', axis=1)
    df = df.fillna('')
    return df


def drop_useless_char_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unnecessary characters in a pandas DataFrame
    :param df: the pandas DataFrame
    :return pandas DataFrame
    """
    df = df.replace('\n', ' ', regex=True)
    df = df.replace('\r', ' ', regex=True)
    df = df.replace('\t', ' ', regex=True)
    return df.apply(
        lambda c: beta_utils.drop_unnecessary_whitespace_in_series(c))
