import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from official import nlp
from official.nlp import bert
import official.nlp.bert.tokenization

import beta_utils

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/3"
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_zh_preprocess/3"
print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_dir = os.path.join(root_dir, 'BetaData')

# Load intent classes
saved_class_path = './intent_classes.txt'
with open(saved_class_path, encoding='utf-8') as f:
    intent_classes = f.read().strip().split('\n')
print(f'Intent class path: {saved_class_path}')

# Load saved model
saved_model_path = './beta_bert_intent'
saved_model_config = 'bert_zh_L-12_H-768_A-12/3'
intent_model = tf.saved_model.load(saved_model_path)
print(f'Model saved path: {saved_model_path}')
print(f'Model configuration: {saved_model_config}')
bert_vocab = os.path.join(saved_model_path, 'assets', 'vocab.txt')
tokenizer = bert.tokenization.FullTokenizer(vocab_file=bert_vocab)


def inference_colleague_intent():

    print('Inference of queries of large scale by colleagues')
    data_file_str = os.path.join(data_root_dir, 'colleague_intent_input.xlsx')
    df = pd.read_excel(data_file_str, names=['text'], engine='openpyxl')
    print(df)

    intent_indices = tf.argsort(
        [tf.sigmoid(intent_model([tf.constant(
            beta_utils.replace_token_for_bert(t))]))[0] for t in df['text']],
        axis=1, direction='DESCENDING')
    df['best_label'] = [intent_classes[intent_indices[j][0]] for j in
                        range(len(intent_indices))]
    df['second_best_label'] = [intent_classes[intent_indices[j][1]] for j in
                               range(len(intent_indices))]
    df['third_best_label'] = [intent_classes[intent_indices[j][2]] for j in
                              range(len(intent_indices))]
    print(df)

    data_file_str = os.path.join(data_root_dir, 'colleague_intent_output.xlsx')
    df.to_excel(data_file_str, header=1, index=False, engine='openpyxl')
    print(f'prediction exported to {data_file_str}')
    print()


def inference_back_translation_intent():

    print('Inference of queries of large scale by back translation')
    data_file_str = os.path.join(data_root_dir, '')
    df = pd.read_excel(data_file_str,
                       names=['label', 'text'], engine='openpyxl')
    print(df)

    df['prediction'] = [intent_classes[tf.argmax(
        [tf.sigmoid(intent_model([tf.constant(
            beta_utils.replace_token_for_bert(t))]))[0] for t in df['text']])]]


def inference_from_console():

    print('Inference from console input')
    try:
        size = int(input())
    except ValueError:
        size = 0
    for i in range(size):
        print(f'Input {i}: ', end='')
        text = beta_utils.replace_token_for_bert(input())
        result = tf.sigmoid(intent_model(tf.constant([text])))
        print(f'Class: {intent_classes[tf.argmax(result[0])]}')


def inference_from_examples():

    print('Inference from given examples')
    examples = [
        '什么是ETF基金',
        '什么是FOF基金',
        '什么是LOF基金',
        '什么是MOM基金',
        '什么是QDII基金',
        '什么是REITS',
        'ETF基金有几种分别是啥',
        '天弘',
        '易方达',
        '华夏大盘',
        'Kevin zhou',
        '004549 009999',
    ]
    intent_model(tf.constant(['0']))
    for example in examples:
        print(f'Example: {example}')
        text = beta_utils.replace_token_for_bert(example)
        result = tf.sigmoid(intent_model(tf.constant([text])))
        print(f'Class:   {intent_classes[tf.argmax(result[0])]}')
        print()


inference_from_examples()
