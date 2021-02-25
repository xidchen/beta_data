# Setup

import collections
import json
import os
import re
import spacy
from spacy.tokens import Doc
from spacy.training import Alignment, offsets_to_biluo_tags
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import tokenizers
from official.nlp import bert, optimization

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')
tf.constant(0)
print()


# NER

# Load spaCy and BERT model
nlp = spacy.load('zh_core_web_trf')
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
data_file_str = os.path.join(data_root_dir, 'demo_beta_ner_1.jsonl')

original_sentences = []
with open(data_file_str, encoding='utf-8') as f:
    for line in f:
        original_sentences.append(json.loads(line))
print(*original_sentences, sep='\n')
print()


# Split long sentences
"""Split text at sentence terminating punctuations, 
and at Chinese commas if text length over 120, 
record offsets, and finally store json-formatted data in a list"""
print('Split long sentences')
comma = '，'
max_length = 120


def split_long_sentence(_sentence: str, _max_len: int) -> []:
    """Split sentence at Chinese comma if length over max length,
    and keep an overlap as long as possible to maintain coherence"""
    if len(_sentence) <= _max_len:
        return [_sentence]
    res = []
    subs = _sentence.split(comma)
    subs[:-1] = [sub + comma for sub in subs[:-1]]
    cur_sent = collections.deque()
    cur_len = 0
    for i in range(len(subs)):
        cur_len += len(subs[i])
        if cur_len >= _max_len and cur_sent:
            res.append(''.join(cur_sent))
        cur_sent.append(subs[i])
        while cur_len > _max_len:
            cur_len -= len(cur_sent.popleft())
    if cur_sent:
        res.append(''.join(cur_sent))
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
    doc = nlp(sentence_json["text"])
    original_labels = sentence_json["labels"]
    lsid = sentence_json["id"]
    for sentence in doc.sents:
        sentence = str(sentence)
        for short_sentence in split_long_sentence(sentence, max_length):
            span_start = sentence_json["text"].find(short_sentence)
            span_end = span_start + len(short_sentence)
            original_span = (span_start, span_end)
            adjusted_labels = adjust_label_offset(original_labels,
                                                  original_span)
            short_sentence_json = {"id": str(lsid) + "-" + str(ssid),
                                   "text": short_sentence,
                                   "labels": adjusted_labels,
                                   "original_span": original_span}
            short_sentences.append(short_sentence_json)
            ssid += 1
print(*short_sentences, sep='\n')
print()


# BERT tokenizer
"""Run spaCy tokenizer with BertWordPieceTokenizer"""
print('BERT tokenizer')


class BertTokenizer:
    def __init__(self, vocab, vocab_file, lowercase=True):
        self.vocab = vocab
        self._tokenizer = tokenizers.BertWordPieceTokenizer(
            vocab=vocab_file, lowercase=lowercase)

    def __call__(self, _text):
        _tokens = self._tokenizer.encode(_text)
        words, spaces = [], []
        for i, (_text, (start, end)) in enumerate(
                zip(_tokens.tokens, _tokens.offsets)):
            words.append(_text)
            if i < len(_tokens.tokens) - 1:
                # If next start != current end we assume a space in between
                next_start, next_end = _tokens.offsets[i + 1]
                spaces.append(next_start > end)
            else:
                spaces.append(True)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp.tokenizer = BertTokenizer(vocab=nlp.vocab, vocab_file=bert_vocab)
print('Tokenzier : BertWordPieceTokenizer')
print()


# NER BILUO tagging
"""BERT NER BILUO tagging"""
print('NER BILUO tagging')


def replace_token_for_bert(_text: str) -> str:
    replace_dict = {'“': '"',
                    '”': '"',
                    '‘': '\'',
                    '’': '\''}
    for k, v in replace_dict.items():
        _text = _text.replace(k, v)
    return _text.lower()


def bert_biluo_tagging(_text: str, _entities: [], _tokens: []) -> []:
    """BERT NER BILUO tagging"""
    res = []
    whitespaces = [i for i, _t in enumerate(_text) if _t == ' ']
    _t_start = 0

    def strip_whitespace(_ents: []) -> []:
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


for sentence_json in short_sentences:
    text = replace_token_for_bert(sentence_json["text"])
    tokens = [token.text for token in nlp(text)][1:-1]
    print(f'Tokens: {tokens}')
    ner_tags = bert_biluo_tagging(text, sentence_json["labels"], tokens)
    print(f'Tags:   {list(zip(tokens, ner_tags))}')
    print()

# # Extra testing
# s = '上海NLP Engineer招聘'
# labels = [(2, 14, 'Job')]
# text = replace_token_for_bert(s)
# tokens = [token.text for token in nlp(s)][1:-1]
# print(f'Tokens: {tokens}')
# ner_tags = bert_biluo_tagging(s, labels, tokens)
# print(f'Tags:   {list(zip(tokens, ner_tags))}')
# print()


# Split train, valid and test dataset
"""Split in memory"""

# Augment train and valid dataset
"""Augment in memory"""

# Check everything is right


# Training


# Prediction
