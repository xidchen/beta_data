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
    s1, s2, lb, res = '', '', '', {}
    if flask.request.method == 'GET':
        s1 = flask.request.args.get('s1', '')
        s2 = flask.request.args.get('s2', '')
        lb = flask.request.args.get('lb', '')
    if flask.request.method == 'POST':
        s1 = flask.request.form.get('s1', '')
        s2 = flask.request.form.get('s2', '')
        lb = flask.request.form.get('lb', '')
    print(f"Title: {s1}")
    print(f"Article: {s2}")
    if lb:
        try:
            lb = eval(lb)
            assert isinstance(lb, list) is True
            print(f"Label {len(lb)}: {lb}")
            print("Title and article will not be considered.")
            res['status'] = '200 OK'
            res['l_embed'] = useml_embed(lb).numpy().tolist()
        except (NameError, SyntaxError):
            print(f"Label parsing error: {lb}")
            res['status'] = '400 Bad Request'
        return res
    ml = 1000
    if s1 or s2:
        res['status'] = '200 OK'
        s1 = beta_utils.split_one_line_long_article(s1, ml) if s1 else []
        s2 = beta_utils.split_one_line_long_article(s2, ml) if s2 else []
        q = s1 + s2 if s1 and s2 else s1 if s1 else s2
        q_embed = useml_embed(q)
        res['s1_embed'] = tf.reduce_mean(
            q_embed[:len(s1)], 0).numpy().tolist() if s1 else []
        res['s2_embed'] = tf.reduce_mean(
            q_embed[len(s1):], 0).numpy().tolist() if s2 else []
    else:
        res['status'] = '400 Bad Request'
        res['s1_embed'] = res['s2_embed'] = []
    print(f"s1_embed[:6]: {res['s1_embed'][:6]}")
    print(f"s2_embed[:6]: {res['s2_embed'][:6]}")
    return flask.jsonify(res)


if __name__ == '__main__':
    useml_embed = hub.load("https://tfhub.dev/google/"
                           "universal-sentence-encoder-multilingual-large/3")
    app.run('0.0.0.0', 5100)
