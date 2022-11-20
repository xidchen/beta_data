import flask
import tensorflow as tf
import tensorflow_text

import beta_utils

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


intent_class_path = './intent_classes.txt'
intent_model_path = './SavedModels/beta_bert_intent_l367_t9670_e8_f88_sm'
bert_model_config = 'bert_zh_L-12_H-768_A-12/4'
with open(intent_class_path, encoding='utf-8') as f:
    intent_classes = f.read().strip().split('\n')
print(f'Intents size: {len(intent_classes)}')
print(f'Intent model path: {intent_model_path}')
print(f'BERT model configuration: {bert_model_config}')
intent_model = tf.saved_model.load(intent_model_path)
intent_model(tf.constant(['0']))
print('Intent model loaded')


app = flask.Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    q = flask.request.form.get('q', '').strip()
    print(f'Q: {q}')
    res = {}
    if q:
        res['status'] = '200 OK'
        q = beta_utils.replace_token_for_bert(q)
        res['intent'] = intent_classes[tf.argmax(
            tf.sigmoid(intent_model(tf.constant([q])))[0])]
    else:
        res['status'] = '400 Bad Request'
    print(res)
    return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5410)
