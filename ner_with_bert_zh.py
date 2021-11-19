# Setup

import abc
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

import beta_bert
import beta_utils

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
max_length = 120


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
    for sentence in beta_utils.split_paragraph(original_text, max_length):
        for short_sentence in beta_utils.split_long_sentence(
                sentence, max_length):
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


raw_ds = []
for sentence_json in short_sentences:
    text = beta_bert.replace_token_for_bert(sentence_json["text"])
    tokens = [token for token in tokenizer.tokenize(text)]
    ner_tags = beta_bert.ner_entity_to_tagging(
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
