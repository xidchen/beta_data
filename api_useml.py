import flask
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

import beta_utils

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    t, a, lb = None, None, None
    if flask.request.method == 'GET':
        t = flask.request.args.get('t', '')
        a = flask.request.args.get('a', '')
        lb = flask.request.args.get('lb', '')
    if flask.request.method == 'POST':
        t = flask.request.form.get('t', '')
        a = flask.request.form.get('a', '')
        lb = flask.request.form.get('lb', '')
    print(f"Title: {t}")
    print(f"Article: {a}")
    res = {}
    if lb:
        try:
            lb = eval(lb)
            assert isinstance(lb, list) is True
            print(f"Label {len(lb)}: {lb}")
            print("Title and article will not be considered.")
            res['status'] = '200 OK'
            res['l_embed'] = useml_embed(lb).numpy().tolist()
        except SyntaxError or NameError:
            print(f"Label parsing error: {lb}")
            res['status'] = '400 Bad Request'
        res['t_embed'] = res['a_embed'] = res['q_embed'] = []
        return res
    ml = 1000
    if t or a:
        res['status'] = '200 OK'
        t = beta_utils.split_one_line_long_article(t, ml) if t else []
        a = beta_utils.split_one_line_long_article(a, ml) if a else []
        q = t + a if t and a else t if t else a
        q_embed = useml_embed(q)
        res['t_embed'] = tf.reduce_mean(
            q_embed[:len(t)], 0).numpy().tolist() if t else []
        res['a_embed'] = tf.reduce_mean(
            q_embed[len(t):], 0).numpy().tolist() if a else []
    else:
        res['status'] = '400 Bad Request'
        res['t_embed'] = res['a_embed'] = res['q_embed'] = []
    print(f"t_embed[:6]: {res['t_embed'][:6]}")
    print(f"a_embed[:6]: {res['a_embed'][:6]}")
    return flask.jsonify(res)


if __name__ == '__main__':
    useml_embed = hub.load("https://tfhub.dev/google/"
                           "universal-sentence-encoder-multilingual-large/3")
    app.run('0.0.0.0', 5100)
