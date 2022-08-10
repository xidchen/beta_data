import flask
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

import beta_utils

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


useml = hub.load('https://tfhub.dev/google/'
                 'universal-sentence-encoder-multilingual-large/3')
useml(['0'])


app = flask.Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    s0 = flask.request.form.get('s0', '')
    s1 = flask.request.form.get('s1', '')
    s2 = flask.request.form.get('s2', '')
    if s0:
        try:
            s0 = eval(s0.replace('\r', '').replace('\n', '').strip())
            assert isinstance(s0, list) is True
            print(f'S0 (len: {len(s0)}): {s0}')
            res = {'s0_embed': useml(s0).numpy().tolist()}
        except (NameError, SyntaxError):
            print(f'S0 parsing error: {s0}')
            res = {'error_msg': 'input error'}
        return flask.jsonify(res)
    if s1 or s2:
        ml = 1000
        s1 = s1.replace('\r', '').replace('\n', '').strip()
        s2 = s2.replace('\r', '').replace('\n', '').strip()
        print(f'S1: {s1}')
        print(f'S2: {s2}')
        s1 = beta_utils.split_one_line_long_article(s1, ml) if s1 else []
        s2 = beta_utils.split_one_line_long_article(s2, ml) if s2 else []
        q = s1 + s2 if s1 and s2 else s1 if s1 else s2
        q_embed = useml(q)
        res = {'s1_embed': tf.reduce_mean(
            q_embed[:len(s1)], 0).numpy().tolist() if s1 else [],
               's2_embed': tf.reduce_mean(
            q_embed[len(s1):], 0).numpy().tolist() if s2 else []}
    else:
        res = {'error_msg': 'input error'}
    return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5300)
