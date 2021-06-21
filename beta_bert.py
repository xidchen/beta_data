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


def predict_entities_from_query(_ner, _query: str, _label_map: dict, _tokenizer,
                                _max_seq_len: int, _scheme: str) -> [
    [int, int, str]]:
    """Use BERT NER model to predict entities from a query
    _ner: loaded BERT NER model, in SavedModel format
    _query: input query
    _label_map: mapping dict of label id and name for NER model
    _tokenizer: BERT tokenizer
    _max_seq_len: maximum effective sequence length
    _scheme: tagging scheme ('IO', 'IOB', 'BILUO')
    """
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
    _input_ids = tf.Variable([_input_ids], dtype=tf.int32)
    _input_mask = tf.Variable([_input_mask], dtype=tf.int32)
    _segment_ids = tf.Variable([_segment_ids], dtype=tf.int32)
    _training = tf.constant(False)
    _logits = _ner([_input_ids, _input_mask, _segment_ids, _training])
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


def ner_entity_to_tagging(_text: str,
                          _entities: [[int, int, str]],
                          _tokens: [str],
                          _scheme: str) -> [str]:
    """BERT NER, transform offsets to tagging
    _text: the original text
    _entities: the start offset, end offset, and name of entities
    _tokens: BERT tokenized tokens
    _scheme: tagging scheme ('IO', 'IOB', 'BILUO')
    """
    res = []
    whitespaces = [i for i, _t in enumerate(_text) if _t == ' ']
    _entities = [list(_entity) for _entity in _entities]
    _e = sorted(strip_whitespace(_entities, whitespaces))
    _t_start = 0
    while whitespaces and _t_start == whitespaces[0]:
        _t_start += 1
        whitespaces.pop(0)

    def extend_end(_end: int, _idx: int) -> int:
        """Return end position of continuous token given that of current one
        _end: end position of current token
        _idx: index of current token"""
        for _i in range(_idx + 1, len(_tokens)):
            if _tokens[_i].startswith('##'):
                _end += len(_tokens[_i][2:])
            else:
                break
        return _end

    for i, _t in enumerate(_tokens):
        if _t.startswith('##'):
            _t_end = _t_start + len(_t[2:])
            res.append('X')
        else:
            if _t == '[UNK]':
                _t_end = _t_start + 1
            else:
                _t_end = _t_start + len(_t)
            if _scheme == 'IO':
                if not _e or _t_start < _e[0][0]:
                    res.append('O')
                elif _t_start >= _e[0][0] and extend_end(_t_end, i) < _e[0][1]:
                    res.append('I-' + _e[0][2])
                elif _t_start >= _e[0][0] and extend_end(_t_end, i) == _e[0][1]:
                    res.append('I-' + _e[0][2])
                    _e.pop(0)
            if _scheme == 'IOB':
                if not _e or _t_start < _e[0][0]:
                    res.append('O')
                elif _t_start == _e[0][0] and extend_end(_t_end, i) < _e[0][1]:
                    res.append('B-' + _e[0][2])
                elif _t_start > _e[0][0] and extend_end(_t_end, i) < _e[0][1]:
                    res.append('I-' + _e[0][2])
                elif _t_start >= _e[0][0] and extend_end(_t_end, i) == _e[0][1]:
                    res.append('I-' + _e[0][2])
                    _e.pop(0)
            if _scheme == 'BILUO':
                if not _e or _t_start < _e[0][0]:
                    res.append('O')
                elif _t_start == _e[0][0] and extend_end(_t_end, i) < _e[0][1]:
                    res.append('B-' + _e[0][2])
                elif _t_start > _e[0][0] and extend_end(_t_end, i) < _e[0][1]:
                    res.append('I-' + _e[0][2])
                elif _t_start > _e[0][0] and extend_end(_t_end, i) == _e[0][1]:
                    res.append('L-' + _e[0][2])
                    _e.pop(0)
                elif _t_start == _e[0][0] and extend_end(_t_end, i) == _e[0][1]:
                    res.append('U-' + _e[0][2])
                    _e.pop(0)
        while whitespaces and _t_end == whitespaces[0]:
            _t_end += 1
            whitespaces.pop(0)
        _t_start = _t_end
    return res


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
                                    _e[1] = _idx + len(_tokens[_i][2:])
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
