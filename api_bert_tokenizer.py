import flask
import official.nlp.bert.tokenization as onbt
import os
import tensorflow_hub as hub

import beta_utils


app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    s, res = '', []
    if flask.request.method == 'GET':
        s = flask.request.args.get('s', '')
    if flask.request.method == 'POST':
        s = flask.request.form.get('s', '')
    if s:
        s = beta_utils.replace_token_for_bert(s)
        res = tokenizer.tokenize(s)
    return flask.jsonify(res)


if __name__ == '__main__':
    url = 'https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4'
    vocab = os.path.join(hub.resolve(url), 'assets', 'vocab.txt')
    tokenizer = onbt.FullTokenizer(vocab_file=vocab)
    app.run('0.0.0.0', 5400)
