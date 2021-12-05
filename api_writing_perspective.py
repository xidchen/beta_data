import flask
import json
import requests
import tensorflow as tf

import beta_code

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    t, a = None, None
    if flask.request.method == 'GET':
        t = flask.request.args.get('t', '')
        a = flask.request.args.get('a', '')
    if flask.request.method == 'POST':
        t = flask.request.form.get('t', '')
        a = flask.request.form.get('a', '')
    res = {}
    top_count = 1
    p_t = .5
    p_a = 1 - p_t
    if t or a:
        res['status'] = '200 OK'
        _r = requests.post(url, data={'t': t, 'a': a})
        t_embed = json.loads(_r.text)['t_embed']
        a_embed = json.loads(_r.text)['a_embed']
        q_embed = [p_t * _t + p_a * _a for (_t, _a) in zip(t_embed, a_embed)
                   ] if t and a else t_embed if t else a_embed
        q_embed = tf.reshape(tf.constant(q_embed), [1, len(q_embed)])
        m = tf.einsum('ij,kj->ik', q_embed, wp_embeds)
        wp_sim_tuples = sorted([*zip(wp_names, m[0].numpy())],
                               key=lambda i: i[1], reverse=True)
        res['perspectives'] = [{
            'id': wp_name_to_id.get(wp_name, ''), 'name': wp_name,
            'similarity': round(float(wp_sim), 4)} for (wp_name, wp_sim)
                                  in wp_sim_tuples][:top_count]
    else:
        res['status'] = '400 Bad Request'
        res['perspectives'] = []
    print(res)
    return flask.jsonify(res)


if __name__ == '__main__':
    url = 'http://172.17.12.57:5100'
    wp_name_and_id, wp_to_event = beta_code.get_perspective_and_event([0, 1])
    wp_name_to_id, wp_id_to_name = wp_name_and_id
    wp_names = [wp_name for wp_name in wp_id_to_name.values()]
    r = requests.post(url, data={'lb': str(wp_names)})
    wp_embeds = tf.constant(json.loads(r.text)['l_embed'])
    app.run('0.0.0.0', 5300)
