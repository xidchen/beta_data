import flask
import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from official.nlp import bert
import official.nlp.bert.tokenization

import beta_bert
import beta_code
import beta_utils

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    q, v = None, None
    if flask.request.method == 'GET':
        q = flask.request.args.get('q')
        v = flask.request.args.get('v')
    if flask.request.method == 'POST':
        q = flask.request.form.get('q')
        v = flask.request.form.get('v')
    res = {}
    if q is not None:
        res['status'] = '200 OK'
        q = beta_utils.replace_token_for_bert(str(q).strip())
        intent = intent_classes[tf.argmax(
            tf.sigmoid(intent_model(tf.constant([q])))[0])]
        res['intent'] = {'id': i_name_to_id.get(intent, ''),
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
            e_text = q[entity[0]:entity[1]]
            e_type = entity[2]
            e_code = e_name_to_code.get(e_type, {}).get(e_text, '')
            if isinstance(e_code, str):
                if v is None:
                    res['entities'].append(
                        {'code': e_code, 'text': e_text, 'type': e_type})
                else:
                    e_name = e_code_to_name.get(e_type, {}).get(e_code, '')
                    res['entities'].append(
                        {'code': e_code, 'text': e_text, 'type': e_type,
                         'name': e_name})
            if isinstance(e_code, list):
                e_code_list = e_code
                for e_code in e_code_list:
                    if v is None:
                        res['entities'].append(
                            {'code': e_code, 'text': e_text, 'type': e_type})
                    else:
                        e_name = e_code_to_name.get(e_type, {}).get(e_code, '')
                        res['entities'].append(
                            {'code': e_code, 'text': e_text, 'type': e_type,
                             'name': e_name})

    else:
        res['status'] = '400 Bad Request'
        res['intent'] = {'name': '', 'id': ''}
        res['entities'] = [{'code': '', 'text': '', 'name': '', 'type': '', }]
    return flask.jsonify(res)


if __name__ == '__main__':
    intent_class_path = './intent_classes.txt'
    intent_model_path = './beta_bert_intent_l323_t7000_e6_f85_sm'
    entity_class_path = './entity_classes.txt'
    entity_model_path = './beta_bert_entity_l7_t920_e3_bio_s1_h5'
    bert_model_config = 'bert_zh_L-12_H-768_A-12/3'
    ner_tagging_scheme = 'BIO'
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
    print(f'BERT model configuration: {bert_model_config}')
    ner_labels = beta_bert.get_ner_labels(_base_labels=entity_classes,
                                          _scheme=ner_tagging_scheme)
    num_labels = len(ner_labels) + 1
    label_map = {i: label for i, label in enumerate(ner_labels, 1)}
    print(f'Label map: {label_map}')
    intent_model = tf.saved_model.load(intent_model_path)
    intent_model(tf.constant(['0']))
    print('Intent model loaded')
    entity_model = beta_bert.load_ner(_model_dir=entity_model_path,
                                      _num_labels=num_labels,
                                      _max_seq_len=max_seq_len)
    print('Entity model loaded')
    i_name_to_id, _ = beta_code.get_intent_code()
    print('Intent code loaded')
    e_name_to_code, e_code_to_name = {}, {}
    for e in entity_classes:
        try:
            e_name_to_code[e], e_code_to_name[e] = beta_code.get_entity_code(e)
            print(f'Entity {e} code loaded')
        except KeyError:
            print(f'Entity {e} code not loaded')
            continue
    app.run('0.0.0.0')
