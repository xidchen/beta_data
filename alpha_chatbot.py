import json
import numpy as np
import os
import pandas as pd
import requests
import time
import werkzeug.datastructures as wd

import beta_code


XLSX_EXTENTION = '.xlsx'


def load_excel(file_path: str, sheet_name: str) -> pd.DataFrame:
    """Load Excel from designated path
    :param file_path: the path of the Excel file
    :param sheet_name: the name of the sheet
    :return: the DataFrame of the Excel file
    """
    return pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')


def extract_kt_and_sq(df: pd.DataFrame) -> pd.DataFrame:
    """Extract kt and sq from certain columns
    :param df: the DataFrame
    :return: the DataFrame of kt and sq
    """
    kt, sq = 'label', 'text'
    df_l = df.assign(kt=df[kt]).assign(sq=df[sq])[['kt', 'sq']]
    df_r = df.assign(kt=df[kt])[['kt']].drop_duplicates()
    df_r = df_r.assign(sq=df_r['kt'])
    df = pd.concat([df_l, df_r])
    return df[['kt', 'sq']].drop_duplicates()


def run_sentence_encoder_on_df(df: pd.DataFrame, hd: bool) -> [[float]]:
    """Run sentence encoder on sq column
    :param df: the DataFrame of kt and sq
    :param hd: whether it is a hot deployment
    :return: the sentence embeddings
    """
    time.sleep(0) if hd else time.sleep(16)
    return run_sentence_encoder(df['sq'].tolist())


def run_sentence_encoder(los: [str]) -> [float]:
    """Run sentence encoder of list of sentences
    :param los: list of sentences
    :return: the embeddings
    """
    u = 'http://localhost:5300'
    return json.loads(requests.post(u, data={'s0': str(los)}).text)['s0_embed']


def get_kt_code() -> {}:
    """Fetch intent API, get mapping of kt name and id
    :return: the mapping of kt name and id
    """
    kt_name_to_id, _ = beta_code.get_intent_code()
    return kt_name_to_id


def allowed_file(name: str) -> bool:
    """Check whether the file name has the allowed extension
    :param name: file name
    :return: True or False
    """
    return name.lower().endswith(XLSX_EXTENTION)


def saved_excel_path(file: wd.FileStorage, directory: str) -> str:
    """Save Excel file in the request and return the path
    :param file: Excel file in the request
    :param directory: directory path to save the Excel file
    :return: the path of the Excel file
    """
    file_name = str(int(time.time())) + XLSX_EXTENTION
    file_path = os.path.join(directory, file_name)
    file.save(file_path)
    return file_path


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
    """Collect top kt in a DataFrame
    :param df: the DataFrame of kt and sq
    :param ss: the similarity scores of se and qe
    :param th: the minimum threshold of similarity
    :return: the DataFrame of kt, sq and ss
    """
    decimals = 4
    df['ss'] = np.around(np.ravel(ss), decimals)
    df = df.where(df['ss'] >= th)
    df = df.sort_values('ss', ascending=False).dropna().drop_duplicates('kt')
    return df.reset_index(drop=True)


def organize_result(df: pd.DataFrame, kt_name_to_id: {}) -> {}:
    """Organize the result from the DataFrame
    :param df: the DataFrame of kt, sq and ss
    :param kt_name_to_id: the mapping of kt name and id
    :return: the dict of intent
    """
    intent = df['kt'][0] if len(df.index) else '[无效意图]'
    return {'intent': {'name': intent, 'id': kt_name_to_id.get(intent, '')},
            'intents': [{k: v for k, v in r.items() if pd.notna(v)}
                        for r in df.to_dict('records')]}


def organize_final_result(res: {}) -> {}:
    """Organize final result for output
    :param res: the result of status, intent and intents
    :return: the result of status and intent
    """
    res.pop('intents')
    return res
