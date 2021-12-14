import flask
import json
import requests
import tensorflow as tf


app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    message = """Welcome to Beta API!
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
    res = json.loads(requests.get(url, params={'q': q}).text)
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
    res = json.loads(requests.get(url, params={'x': x, 'y': y}).text)
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
        r = json.loads(requests.get(url, params={'s1': s}).text)
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
        r = json.loads(requests.get(url, params={'s1': s1, 's2': s2}).text)
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
    res = json.loads(requests.get(url, params={'s': s}).text)
    return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5000)
