import flask
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

import beta_utils

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


MAX_LEN = 500
EXCERPT_LEN = 100

usem = hub.load('https://tfhub.dev/google/'
                'universal-sentence-encoder-multilingual/3')
usem(['0'])


app = flask.Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    s1 = flask.request.form.get('s1', '')
    s2 = flask.request.form.get('s2', '')
    if s1 or s2:
        print(f'S1 (len: {len(s1)}): {s1[:EXCERPT_LEN]}')
        print(f'S2 (len: {len(s2)}): {s2[:EXCERPT_LEN]}')
        s1 = beta_utils.split_one_line_long_article(s1, MAX_LEN) if s1 else []
        s2 = beta_utils.split_one_line_long_article(s2, MAX_LEN) if s2 else []
        q = s1 + s2 if s1 and s2 else s1 if s1 else s2
        q_embed = usem(q)
        s1_embed = tf.linalg.normalize(tf.reshape(
            tf.reduce_mean(q_embed[:len(s1)], 0),
            shape=[1, 512]))[0].numpy().tolist() if s1 else []
        s2_embed = tf.linalg.normalize(tf.reshape(
            tf.reduce_mean(q_embed[len(s1):], 0),
            shape=[1, 512]))[0].numpy().tolist() if s2 else []
        res = {'s1_embed': s1_embed, 's2_embed': s2_embed}
    else:
        res = {'error_msg': 'input error'}
    return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5310)
