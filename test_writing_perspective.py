import json
import numpy as np
import os
import pandas as pd
import random
import requests
import shutil


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_dir = os.path.join(root_dir, 'BetaData')

with open(os.path.join(
        data_root_dir, 'wp_eval', 'titles_all.txt'),
        encoding='utf-8') as f:
    titles = f.read().strip().split('\n')
df = pd.DataFrame(data=titles, columns=['title'])
article_dict = {}
for dirpath, dirs, files in os.walk(os.path.join(data_root_dir, 'wp_txt')):
    for file in files:
        if file.endswith('.txt'):
            file = file[:-4]
        if file in titles:
            with open(os.path.join(dirpath, file + '.txt'),
                      encoding='utf-8-sig') as f:
                article_dict[file] = f.read().strip()
df['article'] = [article_dict[title] for title in df['title']]

df_human = pd.read_excel(
    os.path.join(data_root_dir, 'wp_xlsx', 'wp_human_labeled_integrated.xlsx'),
    names=['title', 'perspectives human labeled'], engine='openpyxl')
df_human['perspectives human labeled'] = [
    eval(_pd) for _pd in df_human['perspectives human labeled']]
df = df.merge(df_human, how='left')


def get_p_r_f_col(_df: pd.DataFrame, _c1: str, _c2: str) -> ([], []):
    _ps, _rs, _fs = [], [], []
    for _i in _df.index:
        _pl1, _pl2 = [], []
        for _p in _df.loc[_i, _c1]['perspectives']:
            _pl1.append(_p['name'])
        for _p in _df.loc[_i, _c2]['perspectives']:
            _pl2.append(_p['name'])
        __p = len([_p for _p in _pl2 if _p in _pl1]) / len(_pl2) if _pl2 else 0
        _ps.append(__p)
        __r = len([_p for _p in _pl1 if _p in _pl2]) / len(_pl1) if _pl1 else 0
        _rs.append(__r)
        _f1 = 2 * __p * __r / (__p + __r) if __p and __r else 0
        _fs.append(_f1)
    return _ps, _rs, _fs


url = 'http://172.17.12.57:5100'
df['perspectives considering only title'] = [
    json.loads(requests.post(url, data={'t': title}).text)
    for title in df['title']]
df['p_only_title'], df['r_only_title'], df['f_only_title'] = get_p_r_f_col(
    df, 'perspectives human labeled', 'perspectives considering only title')
df['perspectives considering only article'] = [
    json.loads(requests.post(url, data={'a': article_dict[title]}).text)
    for title in df['title']]
df['p_only_artile'], df['r_only_artile'], df['f_only_article'] = get_p_r_f_col(
    df, 'perspectives human labeled', 'perspectives considering only article')
df['perspectives considering t and a'] = [json.loads(
    requests.post(url, data={'t': title, 'a': article_dict[title]}).text)
    for title in df['title']]
df['p_t_and_a'], df['r_t_and_a'], df['f_t_and_a'] = get_p_r_f_col(
    df, 'perspectives human labeled', 'perspectives considering t and a')

df.to_excel(os.path.join(data_root_dir, 'wp_xlsx', 'wp_evaluation.xlsx'),
            index=None, engine='openpyxl')
