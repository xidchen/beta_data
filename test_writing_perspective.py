import json
import numpy as np
import os
import pandas as pd
import random
import requests
import shutil
import tensorflow as tf

import beta_code

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


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


def get_p_r_f_col(_df: pd.DataFrame, _c1: str, _c2: str) -> ([], [], []):
    """Get precision, recall, f1 score columns"""
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


url_a = 'http://172.17.12.57:5100'
url_b = 'http://172.17.12.57:5300'
pcot = 'perspectives considering only title'
pcoa = 'perspectives considering only article'
pcta = 'perspectives considering t and a'
pot, rot, fot = 'p_only_title', 'r_only_title', 'f_only_title'
poa, roa, foa = 'p_only_artile', 'r_only_artile', 'f_only_article'
pta, rta, fta = 'p_t_and_a', 'r_t_and_a', 'f_t_and_a'
ecot = 'embedding considering only title'
ecoa = 'embedding considering only article'
ecta = 'embedding considering t and a'


def get_embedding(_r: requests.models.Response) -> [float]:
    """Get text embedding from the response of a request"""
    _p_t = 0.5
    _p_a = 1 - _p_t
    _t_e = json.loads(_r.text)['t_embed']
    _a_e = json.loads(_r.text)['a_embed']
    return [_p_t * _t + _p_a * _a for (_t, _a) in zip(_t_e, _a_e)
            ] if _t_e and _a_e else _t_e if _t_e else _a_e


df[pcot] = df.apply(lambda _r: json.loads(requests.post(
    url_b, data={'t': _r['title']}).text), axis=1)
df[pot], df[rot], df[fot] = get_p_r_f_col(df, phl, pcot)
df[pcoa] = df.apply(lambda _r: json.loads(requests.post(
    url_b, data={'a': _r['article']}).text), axis=1)
df[poa], df[roa], df[foa] = get_p_r_f_col(df, phl, pcoa)
df[pcta] = df.apply(lambda _r: json.loads(requests.post(
    url_b, data={'t': _r['title'], 'a': _r['article']}).text), axis=1)
df[pta], df[rta], df[fta] = get_p_r_f_col(df, phl, pcta)

df.loc[len(df.index) + 1, 'title'] = 'Average'
for x in [pot, rot, fot, poa, roa, foa, pta, rta, fta]:
    df.loc[len(df.index), x] = np.mean(df[x][:-1])
df.to_excel(os.path.join(data_root_dir, 'wp_xlsx', 'wp_evaluation.xlsx'),
            index=None, engine='openpyxl', freeze_panes=(1, 1))

df[ecot] = df.apply(lambda _r: get_embedding(requests.post(
    url_a, data={'t': _r['title']})), axis=1)
df[ecoa] = df.apply(lambda _r: get_embedding(requests.post(
    url_a, data={'a': _r['article']})), axis=1)
df[ecta] = df.apply(lambda _r: get_embedding(requests.post(
    url_a, data={'t': _r['title'], 'a': _r['article']})), axis=1)


def get_perspectives(_ts: tf.Tensor, _sr: pd.Series, _tc: int) -> {}:
    """Get the perspectives of a sample from its similarities
    with training samples and the human labeled perspectives
    of the training samples
    :param _ts: a similarity tensor with shape [1, len(training samples)]
    :param _sr: a series of human labels with length len(training samples)
    :param _tc: top (1, 3, 5, etc.) highest-ranked perspectives
    :returns a dictionary of the predicted perspectives of a sample
    """
    _p = 'perspectives'
    _n, _s = 'name', 'similarity'
    _ps = []
    _pd = {}
    # TODO: Optimize the for loop
    for _i in range(len(_sr.index)):
        for _d in _sr[_i][_p]:
            _sa = _d[_s] * _ts[0][_i].numpy().tolist()
            if _d[_n] in _pd:
                _pd[_d[_n]] += _sa
            else:
                _pd[_d[_n]] = _sa
    _ps = sorted([{_n: _k, _s: _v} for _k, _v in _pd.items()],
                 key=lambda _i: _i[_s], reverse=True)
    return {_p: _ps[:_tc]}


ppbs = 'perspectives prediced by similarity'
df_ppbs = pd.DataFrame({'frac': [], pta: [], rta: [], fta: []})

for frac in np.arange(.01, 1, .01):

    test_count = 5
    frac_ppbs = []
    for _ in range(test_count):
        df_train = df.sample(frac=frac)
        df_train_ri = df_train.reset_index()
        df_eval = df.drop(index=df_train.index.values)
        emb_train = tf.constant(np.stack(df_train[ecta].values),
                                dtype=tf.float32)
        top_count = 1
        # TODO: Optimize matrix multiplication
        df_eval[ppbs] = df_eval.apply(lambda _r: get_perspectives(tf.einsum(
            'ij,kj->ik', tf.reshape(tf.constant(_r[ecta]), [1, len(_r[ecta])]),
            emb_train), df_train_ri[phl], top_count), axis=1)
        df_eval[pta], df_eval[rta], df_eval[fta] = get_p_r_f_col(
            df_eval, phl, ppbs)

        frac_ppbs.append([np.round(frac, 4),
                          np.round(np.mean(df_eval[pta]), 4),
                          np.round(np.mean(df_eval[rta]), 4),
                          np.round(np.mean(df_eval[fta]), 4)])

    frac_ppbs_mean = np.round(np.mean(frac_ppbs, axis=0).tolist(), 4)
    df_ppbs.loc[len(df_ppbs.index)] = frac_ppbs_mean
    print(frac_ppbs_mean)
