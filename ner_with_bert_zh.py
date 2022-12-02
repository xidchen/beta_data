# Setup

import abc
import json
import official.nlp.bert.tokenization as onbt
import official.nlp.optimization as ono
import os
import random
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

import beta_bert
import beta_utils

for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_zh_preprocess/3"
tfhub_bert_zh_handle = "https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4"
bert_model = hub.KerasLayer(tfhub_bert_zh_handle)
print(f'BERT model: {tfhub_bert_zh_handle}')
vocab = bert_model.resolved_object.vocab_file.asset_path.numpy().decode('utf-8')
tokenizer = onbt.FullTokenizer(vocab_file=vocab)
print(f'BERT vocab: {vocab}')
print()


"""Read original NER annotation data and store JSON data in a list"""
print('Read data file')
root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_path = os.path.join(root_dir, 'BetaData')
data_file_path = os.path.join(data_root_path, 'demo_beta_ner.jsonl')

original_sentences = []
with open(data_file_path, encoding='utf-8') as f:
    for line in f:
        original_sentences.append(json.loads(line))
print(*original_sentences[:2], sep='\n')
print()


"""Split text at sentence terminating punctuations, 
and at Chinese commas if text length over 120, 
record offsets, and finally store json-formatted data in a list"""
print('Split long sentences')
max_length = 120


short_sentences = []
for sentence_json in original_sentences:
    ssid = 0
    sid = sentence_json['id']
    original_text = sentence_json['text']
    original_labels = sentence_json['labels']
    for sentence in beta_utils.split_paragraph(original_text, max_length):
        for short_sentence in beta_utils.split_long_sentence(
                sentence, max_length):
            span_start = original_text.find(short_sentence)
            span_end = span_start + len(short_sentence)
            span_in_origin = (span_start, span_end)
            adjusted_labels = beta_utils.adjust_label_offset(
                original_labels, span_in_origin)
            short_sentence_json = {'id': str(sid) + '-' + str(ssid),
                                   'text': short_sentence,
                                   'labels': adjusted_labels,
                                   'span_in_origin': span_in_origin}
            short_sentences.append(short_sentence_json)
            ssid += 1
print(*short_sentences[:2], sep='\n')
print()


"""BERT NER tagging"""
scheme = 'BIO'
print(f'BERT NER tagging scheme: {scheme}')


raw_ds = []
for sentence_json in short_sentences:
    text = beta_utils.replace_token_for_bert(sentence_json['text'])
    tokens = [token for token in tokenizer.tokenize(text)]
    ner_tags = beta_bert.ner_entity_to_tagging(
        text, sentence_json['labels'], tokens, scheme)
    raw_ds.append(list(zip(tokens, ner_tags)))
print(*raw_ds[:2], sep='\n')
print()


def collect_tagging_names(dataset: [[(str, str)]]) -> [str]:
    """Collect label categories from dataset
    :param dataset: NER tagging training data
    :return: list of tagging names
    """
    tagging_names = set()
    for token_tags in dataset:
        for token_tag in token_tags:
            tag = token_tag[1]
            if tag[0] in scheme:
                tagging_names.add(tag)
    return sorted(list(tagging_names))


class_names = collect_tagging_names(raw_ds)
print(f'Class names: {len(class_names), class_names}')
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

train_ds_classes = collect_tagging_names(train_ds)
print(f'Training set classes: {len(train_ds_classes), train_ds_classes}')
valid_ds_classes = collect_tagging_names(val_ds)
print(f'Validation set classes: {len(valid_ds_classes), valid_ds_classes}')
test_ds_classes = collect_tagging_names(test_ds)
print(f'Test set classes: {len(test_ds_classes), test_ds_classes}')
print()


class BertNer(tf.keras.Model, abc.ABC):

    def __init__(self, float_type, num_classes):
        super(BertNer, self).__init__()
        text_input = tf.keras.layers.Input(
            shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(
            tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(
            tfhub_bert_zh_handle, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        sequence_output = outputs['sequence_output']
        self.bert = tf.keras.Model(
            inputs=[text_input], outputs=[sequence_output])
        initializer = tf.keras.initializers.TruncatedNormal(stddev=.02)
        self.dropout = tf.keras.layers.Dropout(.1)
        self.classifier = tf.keras.layers.Dense(num_classes,
                                                activation='softmax',
                                                kernel_initializer=initializer,
                                                name='output',
                                                dtype=float_type)


"""Train NER"""

"""Augment in memory"""
train_copy_size = 1
train_ds *= train_copy_size
print(f'Training set size: {len(train_ds)}')
print(f'Validation set size: {len(val_ds)}')
print(f'Test set size: {len(test_ds)}')
print()

"""Label"""
label_list = ['[CLS]', 'O'] + list(class_names) + ['[SEP]']
num_labels = len(label_list) + 1
label_map = {i: label for i, label in enumerate(label_list, 1)}
print(label_map)

"""Define strategy"""
strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')

"""Loss function"""
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = tf.metrics.SparseCategoricalAccuracy()

"""Optimizer"""
epochs = 3
batch_size = 32
init_lr = 3e-5
steps_per_epoch = int(len(train_ds) / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(.1 * num_train_steps)
optimizer = ono.create_optimizer(init_lr,
                                 num_train_steps=num_train_steps,
                                 num_warmup_steps=warmup_steps)

"""Training"""
