import flask
import official.nlp.bert.tokenization as onbt
import os
import tensorflow as tf
import tensorflow_text

import beta_bert
import beta_utils

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


entity_class_path = './entity_classes.txt'
entity_model_path = './SavedModels/beta_bert_entity_l7_t999_e3_bio_s1_h5'
bert_model_config = 'bert_zh_L-12_H-768_A-12/4'
ner_tagging_scheme = 'BIO'
max_seq_len = 128
tokenizer = onbt.FullTokenizer(os.path.join(entity_model_path, 'vocab.txt'))
with open(entity_class_path, encoding='utf-8') as f:
    entity_classes = f.read().strip().split('\n')
print(f'Entities size: {len(entity_classes)}')
print(f'Entity model path: {entity_model_path}')
print(f'Entity tagging scheme: {ner_tagging_scheme}')
print(f'BERT model configuration: {bert_model_config}')
ner_labels = beta_bert.get_ner_labels(_base_labels=entity_classes,
                                      _scheme=ner_tagging_scheme)
num_labels = len(ner_labels) + 1
label_map = {i: label for i, label in enumerate(ner_labels, 1)}
print(f'Label map: {label_map}')
entity_model = beta_bert.load_ner(_model_dir=entity_model_path,
                                  _num_labels=num_labels,
                                  _max_seq_len=max_seq_len)
print('Entity model loaded')


app = flask.Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    q = flask.request.form.get('query', '').strip()
    print(f'Q: {q}')
    res = {}
    if q:
        res['status'] = '200 OK'
        q = beta_utils.replace_token_for_bert(q)
        res['entities'] = beta_bert.predict_entities_from_query(
            _ner_model=entity_model,
            _query=q,
            _label_map=label_map,
            _tokenizer=tokenizer,
            _max_seq_len=max_seq_len,
            _scheme=ner_tagging_scheme)
    else:
        res['status'] = '400 Bad Request'
    print(res)
    return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5420)
