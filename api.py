import flask
import json
import numpy as np
import os
import requests
import tempfile
import werkzeug.utils as wu


app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    message = """
    Welcome to Beta API!

    route                       parameters  example
    /chatbot                    q           ?q=交银新成长混合和交银精选混合
    /ols                        x, y        ?x=[[0,1],[2,3]]&y=[0,1]
    /sentence_encoder           s0          ?s0=['Hi','Hello']
    /semantic_similarity        s1, s2      ?s1=Hi&s2=Hello
    /semantic_similarity_fast   s1, s2      ?s1=Hi&s2=Hello
    /sentence_pair_encoder      s1, s2      ?s1=Hi&s2=Hello
    /bert_tokenizer             s           ?s=I love you
    """
    return message


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    url = 'http://localhost:5100'
    q, v, res = '', '', {}
    if flask.request.method == 'GET':
        q = flask.request.args.get('q', '')
    if flask.request.method == 'POST':
        q = flask.request.form.get('q', '')
    res = json.loads(requests.post(url, data={'q': q}).text)
    return flask.jsonify(res)


@app.route('/gamma_chatbot', methods=['POST'])
def gamma_chatbot():
    url = 'http://localhost:5110'
    q = flask.request.form.get('query', '')
    p = flask.request.form.get('perspective', '')
    t = flask.request.form.get('threshold', '')
    j = flask.request.form.get('json', '')
    x = flask.request.files.get('excel')
    if j:
        res = json.loads(requests.post(url, data={'j': j}).text)
    elif x:
        xn = wu.secure_filename(x.filename)
        xp = os.path.join(tempfile.gettempdir(), xn)
        x.save(xp)
        res = json.loads(requests.post(url, files={'x': open(xp, 'rb')}).text)
        os.remove(xp)
    else:
        res = json.loads(requests.post(url, data={'q': q, 'p': p, 't': t}).text)
    return flask.jsonify(res)


@app.route('/ols', methods=['GET', 'POST'])
def ols():
    url = 'http://localhost:5200'
    x, y, res = None, None, None
    if flask.request.method == 'GET':
        x = flask.request.args.get('x')
        y = flask.request.args.get('y')
    if flask.request.method == 'POST':
        x = flask.request.form.get('x')
        y = flask.request.form.get('y')
    res = json.loads(requests.post(url, data={'x': x, 'y': y}).text)
    return flask.jsonify(res)


@app.route('/sentence_encoder', methods=['GET', 'POST'])
def sentence_encoder():
    url = 'http://localhost:5300'
    s0, res = '', {}
    if flask.request.method == 'GET':
        s0 = flask.request.args.get('s0', '')
    if flask.request.method == 'POST':
        s0 = flask.request.form.get('s0', '')
    s0 = s0.replace('\r', '').replace('\n', '').strip()
    if s0:
        res = json.loads(requests.post(url, data={'s0': s0}).text)
    else:
        res = {'error_msg': 'input error'}
    return flask.jsonify(res)


@app.route('/semantic_similarity', methods=['GET', 'POST'])
def semantic_similarity():
    url = 'http://localhost:5300'
    s1, s2, res = '', '', {}
    if flask.request.method == 'GET':
        s1 = flask.request.args.get('s1', '')
        s2 = flask.request.args.get('s2', '')
    if flask.request.method == 'POST':
        s1 = flask.request.form.get('s1', '')
        s2 = flask.request.form.get('s2', '')
    s1 = s1.replace('\r', '').replace('\n', '').strip()
    s2 = s2.replace('\r', '').replace('\n', '').strip()
    if s1 and s2:
        r = json.loads(requests.post(url, data={'s1': s1, 's2': s2}).text)
        res = round(np.inner(r['s1_embed'], r['s2_embed']).tolist()[0][0], 4)
    else:
        res = 0
    return flask.jsonify(res)


@app.route('/semantic_similarity_fast', methods=['GET', 'POST'])
def semantic_similarity_fast():
    url = 'http://localhost:5310'
    s1, s2, res = '', '', {}
    if flask.request.method == 'GET':
        s1 = flask.request.args.get('s1', '')
        s2 = flask.request.args.get('s2', '')
    if flask.request.method == 'POST':
        s1 = flask.request.form.get('s1', '')
        s2 = flask.request.form.get('s2', '')
    s1 = s1.replace('\r', '').replace('\n', '').strip()
    s2 = s2.replace('\r', '').replace('\n', '').strip()
    if s1 and s2:
        r = json.loads(requests.post(url, data={'s1': s1, 's2': s2}).text)
        res = round(np.inner(r['s1_embed'], r['s2_embed']).tolist()[0][0], 4)
    else:
        res = 0
    return flask.jsonify(res)


@app.route('/sentence_pair_encoder', methods=['GET', 'POST'])
def sentence_pair_encoder():
    url = 'http://localhost:5310'
    s1, s2, res = '', '', {}
    if flask.request.method == 'GET':
        s1 = flask.request.args.get('s1', '')
        s2 = flask.request.args.get('s2', '')
    if flask.request.method == 'POST':
        s1 = flask.request.form.get('s1', '')
        s2 = flask.request.form.get('s2', '')
    s1 = s1.replace('\r', '').replace('\n', '').strip()
    s2 = s2.replace('\r', '').replace('\n', '').strip()
    res = json.loads(requests.post(url, data={'s1': s1, 's2': s2}).text)
    return flask.jsonify(res)


@app.route('/bert_tokenizer', methods=['GET', 'POST'])
def bert_tokenizer():
    url = 'http://localhost:5400'
    s, res = '', None
    if flask.request.method == 'GET':
        s = flask.request.args.get('s', '')
    if flask.request.method == 'POST':
        s = flask.request.form.get('s', '')
    res = json.loads(requests.post(url, data={'s': s}).text)
    return flask.jsonify(res)


@app.route('/ocr', methods=['POST'])
def ocr():
    url = 'http://localhost:5500'
    file = flask.request.files.get('file')
    if file:
        img_file = wu.secure_filename(file.filename)
        img_file_path = os.path.join(tempfile.gettempdir(), img_file)
        file.save(img_file_path)
        files = {'file': open(img_file_path, 'rb')}
        res = json.loads(requests.post(url, files=files).text)
        os.remove(img_file_path)
    else:
        res = {'error_msg': 'image error'}
    return flask.jsonify(res)


@app.route('/coach', methods=['POST'])
def coach():
    url = 'http://localhost:5600'
    v = flask.request.form.get('url', '')
    u = flask.request.form.get('urls', '')
    r = flask.request.form.get('rhetoric', '')
    k = flask.request.form.get('keywords', '')
    s = flask.request.form.get('transcript', '')
    t = flask.request.form.get('transcripts', '')
    r = r.replace('\r', '').replace('\n', '').strip()
    k = k.replace('\r', '').replace('\n', '').strip()
    if u and r and k and t:
        data = {'urls': u, 'rhetoric': r, 'keywords': k, 'transcripts': t}
        res = json.loads(requests.post(url, data=data).text)
    elif v and r and k and s:
        data = {'url': v, 'rhetoric': r, 'keywords': k, 'transcript': s}
        res = json.loads(requests.post(url, data=data).text)
    else:
        res = {'error_msg': 'input error'}
    return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5000)
