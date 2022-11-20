# Setup

import official.nlp.optimization as ono
import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

import beta_utils

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


# Intent recognition

AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'BetaData/train', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)

class_names = raw_train_ds.class_names
intent_class_path = './intent_classes.txt'
with open(intent_class_path, mode='w', encoding='utf-8') as f:
    f.write('\n'.join(class_names))
train_ds = raw_train_ds.prefetch(buffer_size=AUTOTUNE)

val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'BetaData/train', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)

val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'BetaData/test', batch_size=batch_size)

test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

for text_batch, label_batch in train_ds.take(1).cache():
    for i in range(3):
        print(f'Review: {text_batch.numpy()[i].decode("utf-8")}')
        label = label_batch.numpy()[i]
        print(f'Label : {label} ({class_names[label]})')


# Loading models from TensorFlow Hub

tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4"
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_zh_preprocess/3"

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
print()


# Define model

def build_classifier_model():
    text_input = tf.keras.layers.Input(
        shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(
        tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(
        tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(.1)(net)
    net = tf.keras.layers.Dense(len(class_names), name='classifier')(net)
    return tf.keras.Model(text_input, net)


classifier_model = build_classifier_model()


# Model training

# Loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = tf.metrics.SparseCategoricalAccuracy()

# Optimizer
epochs = 8
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(.1 * num_train_steps)

init_lr = 3e-5
optimizer = ono.create_optimizer(init_lr=init_lr,
                                 num_train_steps=num_train_steps,
                                 num_warmup_steps=num_warmup_steps,
                                 optimizer_type='adamw')

# Loading the BERT model and training
classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

print(f'Training model with {tfhub_handle_encoder}')
history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=epochs)

# Evaluate the model
print()
print('Evaluating test data:')
loss, accuracy = classifier_model.evaluate(test_ds)
print()


# Export for inference

print('Saving models:')
saved_model_path = './beta_bert_intent'
classifier_model.save(saved_model_path, include_optimizer=False)
print(f'Model saved to {saved_model_path}')
print()

reloaded_model = tf.saved_model.load(saved_model_path)


# Inference of queries from examples

def print_my_examples(inputs: [str], results: [float]):
    result_for_printing = [
        f'input: {inputs[j]:30} : class: {class_names[tf.argmax(results[j])]}'
        for j in range(len(inputs))]
    print(*result_for_printing, sep='\n')
    print()


examples = [
    '什么是ETF基金',
    '什么是FOF基金',
    '什么是LOF基金',
    '什么是MOM基金',
    '什么是QDII基金',
    '什么是REITS',
    'ETF基金有几种分别是啥',
    'ETF最少能买卖多少',
    'LOF与ETF有啥区别',
    'MOM基金与FOF基金相比有什么优势',
    '太平洋证券',
    '我想了解一下债基',
]

examples = [beta_utils.replace_token_for_bert(example) for example in examples]
reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))
# original_results = tf.sigmoid(classifier_model(tf.constant(examples)))

print('Results from the saved model:')
print_my_examples(examples, reloaded_results)
# print('Results from the model in memory:')
# print_my_examples(examples, original_results)


# Inference of queries of large scales from an Excel file

def inference_queries_from_excel():
    print('Inference of queries of large scales:')
    root_dir = os.path.dirname(os.path.realpath(__file__))
    data_root_dir = os.path.join(root_dir, 'BetaData')
    data_file_str = os.path.join(data_root_dir, 'demo_beta_text.xlsx')
    df = pd.read_excel(data_file_str, names=['text'], engine='openpyxl')
    df['text'] = [beta_utils.replace_token_for_bert(text)
                  for text in df['text']]
    results = tf.sigmoid(
        reloaded_model.predict(
            tf.constant(df['text']), batch_size=batch_size))
    df['label'] = [class_names[tf.argmax(results[j])]
                   for j in range(len(results))]
    print(df)
    data_file_str = os.path.join(data_root_dir, 'mydb.prediction.xlsx')
    df.to_excel(data_file_str, header=1, index=False, engine='openpyxl')
    print(f'prediction exported to {data_file_str}')
    print()


# inference_queries_from_excel()


# Inference of a single query for demo purpose

def predict_class_of_a_query(query: str):
    print('Inference of a query:')
    query = beta_utils.replace_token_for_bert(query)
    result = tf.sigmoid(reloaded_model(tf.constant([query])))
    print(f'input: {query:30} : class: {class_names[tf.argmax(result[0])]}')
    print()


input_query = '什么是FOF基金'
predict_class_of_a_query(input_query)


# Inference from console input

def predict_class_from_console_inputs():
    print('Inference from console inputs:')
    print('how many inputs (a positive integer): ', end='')
    size = int(input())
    for j in range(size):
        print(f'input {j}: ', end='')
        query = input()
        query = beta_utils.replace_token_for_bert(query)
        result = tf.sigmoid(reloaded_model(tf.constant([query])))
        print(f'class: {class_names[tf.argmax(result[0])]}')


# predict_class_from_console_inputs()
