# Setup

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
bert_vocab = bert_model.resolved_object.vocab_file.asset_path.numpy().decode('utf-8')
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


def split_paragraph(_paragraph: str, _max_len: int, _split: bool = False) -> []:
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


def split_long_sentence(_sentence: str, _max_len: int) -> []:
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


def adjust_label_offset(_labels: [[int, int, str]], _span: [int, int]) -> []:
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


def ner_offset_to_tagging(_text: str,
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
    _t_start = 0

    def strip_whitespace(_ents: [[int, int, str]]) -> []:
        """Strip whitespace if annotated entities have any on either ends"""
        for _ent in _ents:
            for _i in range(_ent[0], _ent[1] - 1, 1):
                if _i in whitespaces:
                    _ent[0] += 1
                else:
                    break
            for _i in range(_ent[1] - 1, _ent[0], -1):
                if _i in whitespaces:
                    _ent[1] -= 1
                else:
                    break
        return _ents

    _e = sorted(strip_whitespace(_entities))

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
        if whitespaces and _t_end == whitespaces[0]:
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


def collect_class_names(_ds: [[(str, str)]]) -> set:
    _class_names = set()
    for _token_tags in _ds:
        for _token_tag in _token_tags:
            _tag = _token_tag[1]
            if _tag[0] in 'BILU':
                _class_names.add(_tag)
    return _class_names


class_names = collect_class_names(raw_ds)
print(f'Class names: {class_names}')
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
print(f'Training set classes: {collect_class_names(train_ds)}')
print(f'Validation set classes: {collect_class_names(val_ds)}')
print(f'Test set classes: {collect_class_names(test_ds)}')
print()


# Augment train dataset
"""Augment in memory"""
train_copy_size = 1
train_ds *= train_copy_size
print(f'Training set size: {len(train_ds)}')
print(f'Validation set size: {len(val_ds)}')
print(f'Test set size: {len(test_ds)}')
print()


# Training
"""Train NER"""
class_list = ['O'] + list(class_names) + ['[CLS]', '[SEP]']
print(class_list)

# Define model

# Loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = tf.metrics.SparseCategoricalAccuracy()

# Optimizer
epochs = 5
num_train_steps = len(train_ds) * epochs
num_warmup_steps = int(.1 * num_train_steps)

init_lr = 3e-5
optimizer = nlp.optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')


# Prediction

def ner_tagging_to_offset(_text: str,
                          _tokens: [str],
                          _tagging: [str],
                          _scheme: str) -> [[int, int, str]]:
    """BERT NER, transform tagging to offsets
    _text: the original text
    _tokens: BERT tokenized tokens, with only first sub-token
    _tagging: BERT ner tagging predicted
    _scheme: tagging scheme"""
    res = []
    return res
