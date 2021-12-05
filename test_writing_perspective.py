import json
import numpy as np
import os
import pandas as pd
import random
import requests
import shutil

import beta_code


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

phl = 'perspectives human labeled'
df_human = pd.read_excel(
    os.path.join(data_root_dir, 'wp_xlsx', 'wp_human_labeled_integrated.xlsx'),
    names=['title', phl], engine='openpyxl')
df_human[phl] = [eval(_pd) for _pd in df_human[phl]]
df = df.merge(df_human, how='left')

wp_to_event = beta_code.get_perspective_and_event([0, 1])[1]


def get_p_r_f_col(_df: pd.DataFrame, _c1: str, _c2: str) -> ([], []):
    _to_event = True
    _ps, _rs, _fs = [], [], []
    for _i in _df.index:
        _pl1, _pl2 = [], []
        for _p in _df.loc[_i, _c1]['perspectives']:
            _name = wp_to_event[_p['name']] if _to_event else _p['name']
            _pl1.append(_name)
        for _p in _df.loc[_i, _c2]['perspectives']:
            _name = wp_to_event[_p['name']] if _to_event else _p['name']
            _pl2.append(_name)
        _pl1, _pl2 = list(set(_pl1)), list(set(_pl2))
        __p = len([_p for _p in _pl2 if _p in _pl1]) / len(_pl2) if _pl2 else 0
        _ps.append(__p)
        __r = len([_p for _p in _pl1 if _p in _pl2]) / len(_pl1) if _pl1 else 0
        _rs.append(__r)
        _f1 = 2 * __p * __r / (__p + __r) if __p and __r else 0
        _fs.append(_f1)
    return _ps, _rs, _fs


url = 'http://172.17.12.57:5300'
pcot = 'perspectives considering only title'
pcoa = 'perspectives considering only article'
pcta = 'perspectives considering t and a'
pot, rot, fot = 'p_only_title', 'r_only_title', 'f_only_title'
poa, roa, foa = 'p_only_artile', 'r_only_artile', 'f_only_article'
pta, rta, fta = 'p_t_and_a', 'r_t_and_a', 'f_t_and_a'
df[pcot] = [
    json.loads(requests.post(url, data={'t': title}).text)
    for title in df['title']]
df[pot], df[rot], df[fot] = get_p_r_f_col(df, phl, pcot)
df[pcoa] = [
    json.loads(requests.post(url, data={'a': article_dict[title]}).text)
    for title in df['title']]
df[poa], df[roa], df[foa] = get_p_r_f_col(df, phl, pcoa)
df[pcta] = [json.loads(
    requests.post(url, data={'t': title, 'a': article_dict[title]}).text)
    for title in df['title']]
df[pta], df[rta], df[fta] = get_p_r_f_col(df, phl, pcta)

df.loc[len(df.index) + 1, 'title'] = 'Average'
for x in [pot, rot, fot, poa, roa, foa, pta, rta, fta]:
    df.loc[len(df.index), x] = np.mean(df[x][:-1])
df.to_excel(os.path.join(data_root_dir, 'wp_xlsx', 'wp_evaluation.xlsx'),
            index=None, engine='openpyxl', freeze_panes=(1, 1))
