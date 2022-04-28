import flask
import numpy as np
import re
import tensorflow as tf

import beta_audio

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


app = flask.Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    file_path = flask.request.form.get('file_path')
    if file_path:
        print(f'Path: {file_path}')
        features = np.array(beta_audio.feature_extraction(
            file_path, dims=f_dims, gate=f_gate), ndmin=2)
        prediction = model(tf.constant(features, dtype=tf.float32))
        res = int(tf.tensordot(prediction, grades, axes=1).numpy()[0])
        return flask.jsonify(res)
    else:
        res = {'error_msg': 'input error'}
        return flask.jsonify(res)


if __name__ == '__main__':
    model_path = './SavedModels/fluency_mlp_t1049_g20111_f66_sm'
    g = re.search(r'_g(\d+)_', model_path).group(1)
    f_dims = [int(d) for d in [g[:-3], g[-3:-2], g[-2:-1], g[-1]]]
    f_gate = [1 if d else 0 for d in f_dims]
    grades = [gd * 11. for gd in [1, 3, 5, 7, 9]]
    model = tf.saved_model.load(model_path)
    model(tf.random.normal([1, np.dot(f_dims, f_gate)]))
    print(f'[INFO] Model {model_path} loaded')
    app.run('0.0.0.0', 5610)
