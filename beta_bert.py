import abc
import json
import os
import tensorflow as tf

import bert_modeling


def get_ner_labels(_base_labels: [str], _scheme: str) -> [str]:
    labels = []
    if _scheme == 'IO':
        labels = [tag + '-' + label for label in _base_labels for tag in 'I']
    if _scheme == 'IOB':
        labels = [tag + '-' + label for label in _base_labels for tag in 'IB']
    if _scheme == 'BILUO':
        labels = [tag + '-' + label for label in _base_labels for tag in 'BILU']
    label_names = ["[CLS]", "O"] + labels + ["[SEP]"]
    print(f'Label names: {label_names}')
    return label_names


def load_ner(_model_dir: str, _num_labels: int, _max_seq_len: int):
    _config = json.load(open(os.path.join(_model_dir, "bert_config.json")))
    _ner = BertNer(_config, tf.float32, _num_labels, _max_seq_len)
    _ids = tf.ones((1, 128), dtype=tf.int64)
    _ = _ner(_ids, _ids, _ids, training=False)
    _ner.load_weights(os.path.join(_model_dir, "model.h5"))
    return _ner


class BertNer(tf.keras.Model, abc.ABC):

    def __init__(self,
                 model_config: dict,
                 float_type,
                 num_labels: int,
                 max_seq_length: int,
                 final_layer_initializer=None):
        """
        model_config : string or dict (only dict here)
                    string: bert pretrained model directory
                      with bert_config.json and bert_model.ckpt
                    dict: bert model config, pretrained weights are not restored
        float_type : tf.float32
        num_labels : num of tags in NER task
        max_seq_length : max_seq_length of tokens
        final_layer_initializer : default: tf.keras.initializers.TruncatedNormal
        """
        super(BertNer, self).__init__()

        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,),
                                               dtype=tf.int32,
                                               name='input_word_ids')
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,),
                                           dtype=tf.int32,
                                           name='input_mask')
        input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,),
                                               dtype=tf.int32,
                                               name='input_type_ids')

        bert_config = bert_modeling.BertConfig.from_dict(model_config)

        bert_layer = bert_modeling.BertModel(
            # will change to official.nlp.bert.bert_model
            config=bert_config, float_type=float_type)

        _, sequence_output = bert_layer(input_word_ids,
                                        input_mask,
                                        input_type_ids)

        self.bert = tf.keras.Model(
            inputs=[input_word_ids, input_mask, input_type_ids],
            outputs=[sequence_output])

        initializer = tf.keras.initializers.TruncatedNormal(
            stddev=bert_config.initializer_range) \
            if final_layer_initializer else final_layer_initializer

        self.dropout = tf.keras.layers.Dropout(
            rate=bert_config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            num_labels,
            kernel_initializer=initializer,
            activation='softmax',
            name='output',
            dtype=float_type)

    def call(self,
             input_word_ids,
             input_mask=None,
             input_type_ids=None,
             **kwargs):
        sequence_output = self.bert(
            [input_word_ids, input_mask, input_type_ids], **kwargs)
        sequence_output = self.dropout(sequence_output,
                                       training=kwargs.get('training', False))
        logits = self.classifier(sequence_output)
        return logits


def predict_entities_from_query(_ner_model,
                                _query: str,
                                _label_map: dict,
                                _tokenizer,
                                _max_seq_len: int,
                                _scheme: str) -> [[int, int, str]]:
    _query_transcoded = replace_token_for_bert(_query)
    _token_list = _tokenizer.tokenize(_query_transcoded)
    _tokens = []
    for _token in _token_list:
        if not _token.startswith('##'):
            _tokens.append(_token)
    if len(_tokens) >= _max_seq_len - 1:
        _tokens = _tokens[0:(_max_seq_len - 2)]
    _tokens.insert(0, '[CLS]')
    _tokens.append('[SEP]')
    _input_ids = _tokenizer.convert_tokens_to_ids(_tokens)
    _input_mask = [1] * len(_input_ids)
    _segment_ids = []
    for x in (_input_ids, _input_mask, _segment_ids):
        x.extend([0] * (_max_seq_len - len(x)))
    _input_ids = tf.Variable([_input_ids], dtype=tf.int64)
    _input_mask = tf.Variable([_input_mask], dtype=tf.int64)
    _segment_ids = tf.Variable([_segment_ids], dtype=tf.int64)
    _logits = _ner_model(_input_ids, _input_mask, _segment_ids)
    _logits_label = tf.argmax(_logits, axis=2)
    _logits_label = _logits_label.numpy().tolist()[0]
    _logits_label = _logits_label[1:_tokens.index('[SEP]')]
    _labels = [_label_map[_label] for _label in _logits_label]
    return ner_tagging_to_entity(_query, _token_list, _labels, _scheme)


def replace_token_for_bert(_text: str) -> str:
    replace_dict = {'“': '"',
                    '”': '"',
                    '‘': '\'',
                    '’': '\'',
                    '—': '-'}
    for k, v in replace_dict.items():
        _text = _text.replace(k, v)
    return _text.lower()


def ner_tagging_to_entity(_text: str,
                          _tokens: [str],
                          _tagging: [str],
                          _scheme: str) -> [[int, int, str]]:
    """BERT NER, transform predicted tagging to entity names and types
    _text: the original text
    _tokens: BERT tokenized tokens
    _tagging: predicted NER tagging labels
    _scheme: tagging scheme ('IO', 'IOB', 'BILUO')
    """
    res = []
    whitespaces = [_i for _i, _t in enumerate(_text) if _t == ' ']
    for _i, _t in enumerate(_tokens):
        if _t.startswith('##'):
            _tagging.insert(_i, 'X')
    _e = [0, 0, '']
    _idx = 0
    for _i, _t in enumerate(_tagging):
        if _scheme == 'IO':
            if _i == 0:
                if _t.startswith('I-'):
                    _e = [_idx, 0, _t[2:]]
            else:
                if _t.startswith('I-'):
                    if _tagging[_i - 1].startswith('I-'):
                        if _t != _tagging[_i - 1]:
                            _e[1] = _idx
                            res.append(_e)
                            _e = [_idx, 0, _t[2:]]
                    if _tagging[_i - 1] == 'O':
                        _e = [_idx, 0, _t[2:]]
                    if _tagging[_i - 1] == 'X':
                        for _j in range(_i - 2, -1, -1):
                            if _tagging[_j].startswith('I-'):
                                if _t != _tagging[_j]:
                                    _e[1] = _idx
                                    res.append(_e)
                                    _e = [_idx, 0, _t[2:]]
                                break
                            if _tagging[_j] == 'O':
                                _e = [_idx, 0, _t[2:]]
                                break
                    if _i == len(_tagging) - 1:
                        if _tokens[_i] == '[UNK]':
                            _e[1] = _idx + 1
                        else:
                            _e[1] = _idx + len(_tokens[_i])
                        res.append(_e)
                if _t == 'O':
                    if _tagging[_i - 1].startswith('I-'):
                        _e[1] = _idx
                        res.append(_e)
                        _e = [0, 0, '']
                    if _tagging[_i - 1] == 'X':
                        for _j in range(_i - 2, -1, -1):
                            if _tagging[_j].startswith('I-'):
                                _e[1] = _idx
                                res.append(_e)
                                break
                            if _tagging[_j] == 'O':
                                break
                if _t == 'X':
                    if _i == len(_tagging) - 1:
                        if _tagging[_i - 1].startswith('I-'):
                            _e[1] = _idx + len(_tokens[_i][2:])
                            res.append(_e)
                        if _tagging[_i - 1] == 'X':
                            for _j in range(_i - 2, -1, -1):
                                if _tagging[_j].startswith('I'):
                                    _e[1] = _idx
                                    res.append(_e)
                                    break
                                if _tagging[_j] == 'O':
                                    break
        if _tokens[_i].startswith('##'):
            _idx += len(_tokens[_i][2:])
        elif _tokens[_i] == '[UNK]':
            _idx += 1
        else:
            _idx += len(_tokens[_i])
        while whitespaces and _idx >= whitespaces[0]:
            whitespaces.pop(0)
            _idx += 1
    whitespaces = [_i for _i, _t in enumerate(_text) if _t == ' ']
    res = strip_whitespace(res, whitespaces)
    return res


def strip_whitespace(_ents: [[int, int, str]], _ws: [int]) -> []:
    """Strip whitespace if entities have any on either ends"""
    for _ent in _ents:
        for _i in range(_ent[0], _ent[1] - 1, 1):
            if _i in _ws:
                _ent[0] += 1
            else:
                break
        for _i in range(_ent[1] - 1, _ent[0], -1):
            if _i in _ws:
                _ent[1] -= 1
            else:
                break
    return _ents
