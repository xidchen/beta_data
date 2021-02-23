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


def split_long_sentence(sent: str, max_len: int) -> []:
    """Split sentence at Chinese comma if length over max length,
    and keep an overlap as long as possible to maintain coherence"""
    if len(sent) <= max_len:
        return [sent]
    res = []
    subs = sent.split(comma)
    subs[:-1] = [s + comma for s in subs[:-1]]
    cur_sent = collections.deque()
    cur_len = 0
    for i in range(len(subs)):
        cur_len += len(subs[i])
        if cur_len >= max_len and cur_sent:
            res.append(''.join(cur_sent))
        cur_sent.append(subs[i])
        while cur_len > max_len:
            cur_len -= len(cur_sent.popleft())
    if cur_sent:
        res.append(''.join(cur_sent))
    return res


def adjust_label_offset(labels: [[int, int, str]], span: [int, int]) -> []:
    """Adjust label offsets according to text span in original sentence,
    concretely, select label and adjust its offsets if located in the span."""
    res = []
    for label in labels:
        assert type(label[0]) == int
        assert type(label[1]) == int
        assert type(label[2]) == str
        if span[0] <= label[0] and label[1] <= span[1]:
            res.append([label[0] - span[0], label[1] - span[0], label[2]])
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


# BILUO transformation
"""Transform label offsets in json_formatted data to BILUO, 
and store in tuples in lists"""
print('BILUO transformation')
ds_biluo_list = []
for sentence_json in short_sentences:
    doc = nlp(sentence_json["text"])
    tokens = [token.text for token in doc]
    print(f'Tokens: {tokens}')
    entities = sentence_json["labels"]
    tags = offsets_to_biluo_tags(doc=doc, entities=entities)
    print(f'Tags:   {tags}')
print()


# spaCy with BERT tokenizer
"""Run spaCy tokenizer with BertWordPieceTokenizer"""
print('spaCy with BERT tokenizer')


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
print()


# Align with BERT tokenizer
"""Run TFHub BERT preprocessor, align BILUO lists to BERT ones"""
print('Align with BERT tokenizer')


def preprocess_text_for_bert_tokenizer(_text: str) -> str:
    replace_dict = {'“': '"',
                    '”': '"',
                    '‘': '\'',
                    '’': '\''}
    for k, v in replace_dict.items():
        _text = _text.replace(k, v)
    return _text.lower()


for sentence_json in short_sentences:
    text = preprocess_text_for_bert_tokenizer(sentence_json["text"])
    print(f'Tokens: {[token.text for token in nlp(text)]}')
    text_preprocessed = bert_preprocess_model([sentence_json["text"]])
    print(f'Ids:    {text_preprocessed["input_word_ids"]}')
test_str = preprocess_text_for_bert_tokenizer("Google是个好公司")
print(f'BERT tokens:  {[t.text for t in nlp(test_str)]}')


# Split train, valid and test dataset
"""Split in memory"""

# Augment train and valid dataset
"""Augment in memory"""

# Check everything is right


# Training


# Prediction
