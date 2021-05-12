import flask
import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from official.nlp import bert
import official.nlp.bert.tokenization

import beta_bert
import beta_code


physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')
tf.constant(0)


app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    q = None
    if flask.request.method == 'GET':
        q = flask.request.args.get('q')
    if flask.request.method == 'POST':
        q = flask.request.form.get('q')
    res = {}
    if q is not None:
        res['status'] = '200 OK'
        intent = intent_classes[tf.argmax(
            tf.sigmoid(intent_model([tf.constant(q)]))[0])]
        res['intent'] = {'id': intent_name_to_id.get(intent, ''),
                         'name': intent}
        entities = beta_bert.predict_entities_from_query(
            _ner_model=entity_model,
            _query=q,
            _label_map=label_map,
            _tokenizer=tokenizer,
            _max_seq_len=max_seq_len,
            _scheme=ner_tagging_scheme)
        res['entities'] = []
        for entity in entities:
            e_name = q[entity[0]:entity[1]]
            e_type = entity[2]
            e_code = entity_name_to_code.get(e_type, {}).get(e_name, '')
            res['entities'].append({'code': e_code,
                                    'name': e_name,
                                    'type': e_type})
    else:
        res['status'] = '400 Bad Request'
        res['intent'] = {'name': '', 'id': ''}
        res['entities'] = [{'name': '', 'type': '', 'code': ''}]
    return flask.jsonify(res)


if __name__ == '__main__':
    intent_class_path = './intent_classes.txt'
    intent_model_path = './beta_bert_intent_238'
    entity_class_path = './entity_classes.txt'
    entity_model_path = './beta_bert_entity_c3_t354_e3_f94_s1_h5'
    bert_model_config = 'bert_zh_L-12_H-768_A-12/3'
    ner_tagging_scheme = 'IO'
    max_seq_len = 128
    tokenizer = bert.tokenization.FullTokenizer(
        os.path.join(entity_model_path, 'vocab.txt'))
    with open(intent_class_path, encoding='utf-8') as f:
        intent_classes = f.read().strip().split('\n')
    with open(entity_class_path, encoding='utf-8') as f:
        entity_classes = f.read().strip().split('\n')
    print(f'Intents size: {len(intent_classes)}')
    print(f'Entities size: {len(entity_classes)}')
    print(f'Intent model path: {intent_model_path}')
    print(f'Entity model path: {entity_model_path}')
    print(f'Entity tagging scheme: {ner_tagging_scheme}')
    print(f'BERT Model configuration: {bert_model_config}')
    ner_labels = beta_bert.get_ner_labels(_base_labels=entity_classes,
                                          _scheme=ner_tagging_scheme)
    num_labels = len(ner_labels) + 1
    label_map = {i: label for i, label in enumerate(ner_labels, 1)}
    print(f'Label map: {label_map}')
    intent_model = tf.saved_model.load(intent_model_path)
    intent_model([tf.constant('0')])
    print('Intent model loaded')
    entity_model = beta_bert.load_ner(_model_dir=entity_model_path,
                                      _num_labels=num_labels,
                                      _max_seq_len=max_seq_len)
    print('Entity model loaded')
    intent_name_to_id, intent_id_to_name = beta_code.get_intent_code()
    print('Intent code loaded')
    entity_name_to_code = {}
    for e in entity_classes:
        try:
            entity_name_to_code[e] = beta_code.get_entity_code(e)
        except KeyError:
            continue
    print('Entity code loaded')
    app.run('0.0.0.0')
