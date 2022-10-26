import json
import numpy as np
import os
import pandas as pd
import requests
import time
import werkzeug.datastructures as wd


XLSX_EXTENTION = '.xlsx'


def load_excel(file_path: str, sheet_name: str) -> pd.DataFrame:
    """Load Excel from designated path
    :param file_path: the path of the Excel file
    :param sheet_name: the name of the sheet
    :return: the DataFrame of the Excel file
    """
    return pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')


def load_json(json_str: str) -> pd.DataFrame:
    """Load JSON from JSON string and return a DataFrame
    :param json_str: the JSON string
    :return: the DataFrame of the JSON string
    """
    json_str = transform_json_to_columns_format(json_str)
    return pd.read_json(json_str, orient='columns')


def transform_json_to_columns_format(json_str: str) -> str:
    """Transform JSON in a specific format and return in 'columns' format
    :param json_str: the JSON string in a specific format
    :return: the JSON string in 'columns' format
    """
    return json_str


def extract_kt_and_sq(df: pd.DataFrame) -> pd.DataFrame:
    """Extract kt and sq from certain columns
    :param df: the DataFrame of the Excel file
    :return: the DataFrame of kt and sq
    """
    kt, sq = '知识标题', '相似问法'
    df = df.assign(kt=df[kt])
    df = df.assign(sq=(df[sq] + '\n' + df[kt]).str.split('\n')).explode('sq')
    df['sq'] = df['sq'].fillna(df['kt'])
    return df[['kt', 'sq']].drop_duplicates()


def run_sentence_encoder(los: [str]) -> [float]:
    """Run sentence encoder of list of sentences
    :param los: list of sentences
    :return: the embeddings
    """
    u = 'https://w-1.test.betawm.com/athena/sentence_encoder'
    return json.loads(requests.post(u, data={'s0': str(los)}).text)['s0_embed']


def run_sentence_encoder_on_df(df: pd.DataFrame, hd: bool) -> [[float]]:
    """Run sentence encoder on sq column
    :param df: the DataFrame of kt and sq
    :param hd: whether it is a hot deployment
    :return: the sentence embeddings
    """
    time.sleep(0) if hd else time.sleep(20)
    return run_sentence_encoder(df['sq'].tolist())


def run_sentence_encoder_on_str(s: str) -> [float]:
    """Run sentence encoder on a sentence
    :param s: the sentence
    :return: the embedding
    """
    return run_sentence_encoder([s])


def calculate_similarity_scores(se: [[float]], qe: [[float]]) -> [[float]]:
    """Calculate similarity between sentence embeddings and query embedding
    :param se: sentence embeddings
    :param qe: query embedding
    :return: similarity scores
    """
    return np.inner(se, qe)


def collect_top_kt(df: pd.DataFrame, ss: [[float]], th: float) -> pd.DataFrame:
    """
    Collect top kt in a DataFrame
    :param df: the DataFrame of kt and sq
    :param ss: the similarity scores of se and qe
    :param th: the minimum threshold of similarity
    :return: the DataFrame of kt, sq and ss
    """
    decimals = 4
    df['ss'] = np.around(np.ravel(ss), decimals)
    df = df.where(df['ss'] >= th)
    return df.sort_values('ss', ascending=False).dropna().drop_duplicates('kt')


def join_answer(df_l: pd.DataFrame, df_r: pd.DataFrame, pspt: str) -> {}:
    """Join left DataFrame with right DataFrame
    :param df_l: the DataFrame of kt, sq and ss
    :param df_r: the DataFrame of the Excel file
    :param pspt: the perspectives
    :return: the dict of kt, ss, answers, and else
    """
    kt, sq, ss = '知识标题', '相似问法', '相似度'
    df_l = df_l.rename(columns={'kt': kt, 'sq': sq, 'ss': ss})
    df_r = df_r[[c for c in df_r.columns if c.startswith('答案')]]
    pspt = split_by_semicolon(pspt)
    df_r = df_r[[c for c in df_r.columns for p in pspt if p in c]]
    return {'intents': [{k: v for k, v in r.items() if pd.notna(v)}
            for r in df_l.join(df_r).to_dict('records')]}


def split_by_semicolon(s: str) -> [str]:
    """Split a string by semicolon into list of strings
    :param s: the string
    :return: list of strings
    """
    return [_s.strip() for _s in s.split(';') if _s.strip()]


def allowed_file(name: str) -> bool:
    """
    Check whether the file name has the allowed extension
    :param name: file name
    :return: True or False
    """
    return name.lower().endswith(XLSX_EXTENTION)


def saved_excel_path(file: wd.FileStorage, directory: str) -> str:
    """
    Save Excel file in the request and return the path
    :param file: Excel file in the request
    :param directory: directory path to save the Excel file
    :return: the path of the Excel file
    """
    file_name = str(int(time.time())) + XLSX_EXTENTION
    file_path = os.path.join(directory, file_name)
    file.save(file_path)
    return file_path
