import flask
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    x, y, res = None, None, None
    if flask.request.method == 'GET':
        x = flask.request.args.get('x')
        y = flask.request.args.get('y')
    if flask.request.method == 'POST':
        x = flask.request.form.get('x')
        y = flask.request.form.get('y')
    if x and y:
        x = tf.constant(eval(x), dtype=tf.float64)
        y = tf.constant(eval(y), dtype=tf.float64)
        y = tf.reshape(y, [len(y), 1])
        try:
            res = tf.linalg.inv(tf.transpose(x) @ x) @ tf.transpose(x) @ y
            res = tf.reshape(res, [len(res), ])
            res = [round(r, 8) for r in res.numpy().tolist()]
        except tf.errors.InvalidArgumentError:
            res = 'Matrix Op Error'
    else:
        res = '400 Bad Request'
    return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5200)
