import flask
import numpy as np

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
        x, y = np.array(eval(x)), np.array(eval(y))
        try:
            res = np.linalg.inv(np.transpose(x) @ x) @ np.transpose(x) @ y
            res = [round(r, 8) for r in res.tolist()]
        except numpy.linalg.LinAlgError:
            res = 'Singular Matrix Error'
        except ValueError:
            res = 'Input Error'
    else:
        res = '400 Bad Request'
    return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5200)
