import flask
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

import beta_code
import beta_utils

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
    print(f'Title: {t}')
    print(f'Article: {a}')
    res = {}
    top_count = 1
    ml = 1000
    p_t = .5
    p_a = 1 - p_t
    if t or a:
        res['status'] = '200 OK'
        t = beta_utils.split_one_line_long_article(t, ml) if t else []
        a = beta_utils.split_one_line_long_article(a, ml) if a else []
        q = t + a if t and a else t if t else a
        m = tf.einsum('ij,kj->ik', useml_embed(q), wp_embeds)
        m_partial_reduced = tf.stack(
            [2 * p_t * tf.reduce_mean(m[:len(t)], 0),
             2 * p_a * tf.reduce_mean(m[len(t):], 0)]) if t and a else m
        wp_sim_tuples = sorted([*zip(wp_names, tf.reduce_mean(
            m_partial_reduced, 0).numpy())], key=lambda i: i[1], reverse=True)
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
    useml_embed = hub.load("https://tfhub.dev/google/"
                           "universal-sentence-encoder-multilingual-large/3")
    wp_name_to_id, wp_id_to_name = beta_code.get_perspective_code([0, 1])
    wp_names = [wp_name for wp_name in wp_id_to_name.values()]
    wp_embeds = useml_embed(wp_names)
    app.run('0.0.0.0', 5100)
