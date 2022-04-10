import flask
import json
import os
import requests
import tempfile
import tensorflow as tf
import werkzeug.utils

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    message = """
    Welcome to Beta API!

    route                 parameters  example  
    /chatbot              q           ?q=交银新成长混合和交银精选混合
    /ols                  x, y        ?x=[[0,1],[2,3]]&y=[0,1]
    /semantic_similarity  s1, s2      ?s1=Hi&s2=Hello
    /bert_tokenizer       s           ?s=I love you
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
    s, res = '', {}
    if flask.request.method == 'GET':
        s = flask.request.args.get('s', '')
    if flask.request.method == 'POST':
        s = flask.request.form.get('s', '')
    if s:
        r = json.loads(requests.post(url, data={'s1': s}).text)
        res = r['s1_embed']
    else:
        res = []
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
    if s1 and s2:
        r = json.loads(requests.post(url, data={'s1': s1, 's2': s2}).text)
        t = tf.reshape(tf.constant(r['s1_embed']), shape=[1, 512])
        a = tf.reshape(tf.constant(r['s2_embed']), shape=[1, 512])
        res = round(tf.einsum('ij,kj->ik', t, a).numpy().tolist()[0][0], 6)
    else:
        res = 0
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
        img_file = werkzeug.utils.secure_filename(file.filename)
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
    v = flask.request.form.get('url')
    u = flask.request.form.get('urls')
    r = flask.request.form.get('rhetoric')
    k = flask.request.form.get('keywords')
    s = flask.request.form.get('transcript')
    t = flask.request.form.get('transcripts')
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
