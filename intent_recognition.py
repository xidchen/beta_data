import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import official.nlp.bert.tokenization as onbt

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
tokenizer = onbt.FullTokenizer(vocab_file=bert_vocab)


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
        '富国基金的基金经理',
        '德邦创新资本有限责任公司南京汤山温泉二期',
        '江铜熙金(上海)股权投资',
        '华闻安泰尚孚1号特定多客户资产管理计划',
        '天弘-台州债券',
        '诺安双利',
        '银网信联(芜湖)股权投资合伙企业(有限合伙)',
        '朗实风雅私募证券投资基金',
        '执力资产-宁航1号股权投资基金',
        '中邮定期开放债券型证券投资基金',
        '静实至正私募证券投资基金',
        '久期量和投资',
        '第一创业证券',
        '深圳市世纪盛元',
        '北京金源亨立',
        '共青城独角兽投资管理合伙企业(有限合伙)',
        '福州开发区数为资产',
        '海南新长期私募基金',
        '复星联合团体住院医疗',
        '光大永明意外',
        '上海盛世财富五号b款年金',
        '新光海航交通工具意外(b款)',
        '前海建筑工程人员团体意外',
        '新能源基金',
        '最近行情怎么样',
    ]
    intent_model(tf.constant(['0']))
    for example in examples:
        print(f'Example: {example}')
        text = beta_utils.replace_token_for_bert(example)
        result = tf.sigmoid(intent_model(tf.constant([text])))
        print(f'Class:   {intent_classes[tf.argmax(result[0])]}')
        print()


inference_from_examples()
