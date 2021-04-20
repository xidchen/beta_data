# Setup

import abc
import collections
import json
import os
import random
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from official import nlp
from official.nlp import bert
import official.nlp.optimization
import official.nlp.bert.tokenization

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')
tf.constant(0)
print()


# NER

# Load spaCy and BERT model
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_zh_preprocess/3"
tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/3"
print(f'BERT preprocess model : {tfhub_handle_preprocess}')
print(f'BERT encoder model    : {tfhub_handle_encoder}')
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
bert_model = hub.KerasLayer(tfhub_handle_encoder)
bert_vocab = bert_model.resolved_object.vocab_file.asset_path.numpy(
    ).decode('utf-8')
print(f'BERT vocab file:      : {bert_vocab}')
print()


# Read data file
"""Read original ner annotation data line by line, 
and store json-formatted data in a list"""
print('Read data file')
root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_dir = os.path.join(root_dir, 'BetaData')
data_file_str = os.path.join(data_root_dir, 'demo_beta_ner.jsonl')

original_sentences = []
with open(data_file_str, encoding='utf-8') as f:
    for line in f:
        original_sentences.append(json.loads(line))
print(*original_sentences[:1], sep='\n')
print()


# Split long sentences
"""Split text at sentence terminating punctuations, 
and at Chinese commas if text length over 120, 
record offsets, and finally store json-formatted data in a list"""
print('Split long sentences')
sentence_terminators = ['。', '！', '？']
lower_quotation_mark = '”'
comma = '，'
max_length = 120


def split_paragraph(_paragraph: str,
                    _max_len: int,
                    _split: bool = False) -> []:
    """Split paragraph at sentence terminating punctuations"""
    res = []
    if _split:
        cache = ''
        if len(_paragraph) <= _max_len:
            res.append(_paragraph)
            return res
        for _char in _paragraph:
            if _char in sentence_terminators:
                cache += _char
            else:
                if _char == lower_quotation_mark:
                    if cache[-1] in sentence_terminators:
                        res.append(cache + _char)
                        cache = ''
                    else:
                        cache += _char
                else:
                    if cache and cache[-1] in sentence_terminators:
                        res.append(cache)
                        cache = _char
                    else:
                        cache += _char
    else:
        res.append(_paragraph)
    return res


def split_long_sentence(_sentence: str,
                        _max_len: int) -> []:
    """Split sentence at Chinese comma if length over max length,
    and keep an overlap as long as possible to maintain coherence"""
    res = []
    _subs = _sentence.split(comma)
    _subs[:-1] = [_sub + comma for _sub in _subs[:-1]]
    _cur_sent = collections.deque()
    _cur_len = 0
    for i in range(len(_subs)):
        _cur_len += len(_subs[i])
        if _cur_len >= _max_len and _cur_sent:
            res.append(''.join(_cur_sent))
        _cur_sent.append(_subs[i])
        while _cur_len > _max_len:
            _cur_len -= len(_cur_sent.popleft())
    if _cur_sent:
        res.append(''.join(_cur_sent))
    return res


def adjust_label_offset(_labels: [[int, int, str]],
                        _span: [int, int]) -> []:
    """Adjust label offsets according to text span in original sentence,
    concretely, select label and adjust its offsets if located in the span."""
    res = []
    for _l in _labels:
        assert type(_l[0]) == int
        assert type(_l[1]) == int
        assert type(_l[2]) == str
        if _span[0] <= _l[0] and _l[1] <= _span[1]:
            res.append([_l[0] - _span[0], _l[1] - _span[0], _l[2]])
    return res


short_sentences = []
for sentence_json in original_sentences:
    ssid = 0
    original_text = sentence_json["text"]
    original_labels = sentence_json["labels"]
    lsid = sentence_json["id"]
    for sentence in split_paragraph(original_text, max_length):
        for short_sentence in split_long_sentence(sentence, max_length):
            span_start = original_text.find(short_sentence)
            span_end = span_start + len(short_sentence)
            span_in_origin = (span_start, span_end)
            adjusted_labels = adjust_label_offset(original_labels,
                                                  span_in_origin)
            short_sentence_json = {"id": str(lsid) + "-" + str(ssid),
                                   "text": short_sentence,
                                   "labels": adjusted_labels,
                                   "span_in_origin": span_in_origin}
            short_sentences.append(short_sentence_json)
            ssid += 1
print(*short_sentences[:1], sep='\n')
print()


# BERT tokenizer
"""Run spaCy tokenizer with BertWordPieceTokenizer"""
print('BERT tokenizer')
tokenizer = bert.tokenization.FullTokenizer(vocab_file=bert_vocab)
print('Tokenzier : BertWordPieceTokenizer')
print()


# NER tagging
"""BERT NER tagging"""
scheme = 'IO'
print(f'BERT NER {scheme} tagging')


def replace_token_for_bert(_text: str) -> str:
    replace_dict = {'“': '"',
                    '”': '"',
                    '‘': '\'',
                    '’': '\'',
                    '—': '-'}
    for k, v in replace_dict.items():
        _text = _text.replace(k, v)
    return _text.lower()


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


raw_ds = []
for sentence_json in short_sentences:
    text = replace_token_for_bert(sentence_json["text"])
    tokens = [token for token in tokenizer.tokenize(text)]
    ner_tags = ner_offset_to_tagging(
        text, sentence_json["labels"], tokens, scheme)
    raw_ds.append(list(zip(tokens, ner_tags)))
print(*raw_ds[:1], sep='\n')
print()


# Collect label categories
"""Collect categories of BILU labels from dataset"""


def collect_label_names(_ds: [[(str, str)]]) -> set:
    _label_names = set()
    for _token_tags in _ds:
        for _token_tag in _token_tags:
            _tag = _token_tag[1]
            if _tag[0] in 'BILU':
                _label_names.add(_tag)
    return _label_names


label_names = collect_label_names(raw_ds)
print(f'Class names: {label_names}')
print()


# Split train, valid and test dataset
"""Split in memory"""
print(f'Raw dataset size: {len(raw_ds)}')
random.shuffle(raw_ds)
test_ds_split = 0.1
test_ds_size = int(test_ds_split * len(raw_ds))
test_ds = raw_ds[:test_ds_size]
raw_train_ds = raw_ds[test_ds_size:]
val_ds_split = 0.2
val_ds_size = int(val_ds_split * len(raw_train_ds))
train_ds = raw_train_ds[val_ds_size:]
val_ds = raw_train_ds[:val_ds_size]
print(f'Training set classes: {collect_label_names(train_ds)}')
print(f'Validation set classes: {collect_label_names(val_ds)}')
print(f'Test set classes: {collect_label_names(test_ds)}')
print()


# Define model
bert_model = hub.KerasLayer(tfhub_handle_encoder)


class BertNer(tf.keras.Model, abc.ABC):

    def __init__(self, float_type, num_labels):
        super(BertNer, self).__init__()
        text_input = tf.keras.layers.Input(
            shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(
            tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(
            tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        sequence_output = outputs['sequence_output']
        self.bert = tf.keras.Model(
            inputs=[text_input], outputs=[sequence_output])
        initializer = tf.keras.initializers.TruncatedNormal(stddev=.02)
        self.dropout = tf.keras.layers.Dropout(.1)
        self.classifier = tf.keras.layers.Dense(num_labels,
                                                activation='softmax',
                                                kernel_initializer=initializer,
                                                name='output',
                                                dtype=float_type)


# Training
"""Train NER"""

# Augment train dataset
"""Augment in memory"""
train_copy_size = 1
train_ds *= train_copy_size
print(f'Training set size: {len(train_ds)}')
print(f'Validation set size: {len(val_ds)}')
print(f'Test set size: {len(test_ds)}')
print()

# Label
label_list = ['[CLS]', 'O'] + list(label_names) + ['[SEP]']
label_map = {i: label for i, label in enumerate(label_list, 1)}
print(label_map)

# Define strategy
strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')

# Loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = tf.metrics.SparseCategoricalAccuracy()

# Optimizer
epochs = 3
batch_size = 32
init_lr = 3e-5
steps_per_epoch = int(len(train_ds) / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(.1 * num_train_steps)
optimizer = nlp.optimization.create_optimizer(
    init_lr, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
print(f'Optimizer: {type(optimizer)}')


# Prediction

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
        _idx += len(_tokens[_i][2:]) \
            if _tokens[_i].startswith('##') else len(_tokens[_i])
        while whitespaces and _idx >= whitespaces[0]:
            whitespaces.pop(0)
            _idx += 1
    whitespaces = [_i for _i, _t in enumerate(_text) if _t == ' ']
    res = strip_whitespace(res, whitespaces)
    return res