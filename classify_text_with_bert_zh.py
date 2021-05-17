# Setup

import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from official.nlp import optimization

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
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

val_ds = tf.keras.preprocessing.text_dataset_from_directory('BetaData/train',
    batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)

val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

test_ds = tf.keras.preprocessing.text_dataset_from_directory('BetaData/test',
    batch_size=batch_size)

test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

for text_batch, label_batch in train_ds.take(1).cache():
    for i in range(3):
        print(f'Review: {text_batch.numpy()[i].decode("utf-8")}')
        label = label_batch.numpy()[i]
        print(f'Label : {label} ({class_names[label]})')


# Loading models from TensorFlow Hub

tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/3"
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_zh_preprocess/3"

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')


# The preprocessing model

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

text_test = ['降息了，该买点什么']
print(text_test)

text_preprocessed = bert_preprocess_model(text_test)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')


# Using the BERT model

bert_model = hub.KerasLayer(tfhub_handle_encoder)

bert_results = bert_model(text_preprocessed)

print(f'Loaded BERT: {tfhub_handle_encoder}')
print(f'Pooled Output Shape: {bert_results["pooled_output"].shape}')
print(f'Pooled Output Values: {bert_results["pooled_output"][0, :1]}')
print(f'Sequence Output Shape: {bert_results["sequence_output"].shape}')
print(f'Sequence Output Values: {bert_results["sequence_output"][0, :1, :6]}')
print(f'Encoder Outputs Shape: {bert_results["encoder_outputs"][0].shape}')
for i in range(len(bert_results["encoder_outputs"])):
    print(f'Encoder Block {i} Output Values: '
          f'{bert_results["encoder_outputs"][i][0, :1, :6]}')


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
bert_raw_result = classifier_model(tf.constant(text_test))
print(f'BERT Output Shape: {tf.sigmoid(bert_raw_result).shape}')


# Model training

# Loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = tf.metrics.SparseCategoricalAccuracy()

# Optimizer
epochs = 5
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(.1 * num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
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
classifier_model.save(saved_model_path)
print(f'model saved to {saved_model_path}')
print()

# reloaded_model = tf.saved_model.load(saved_model_path)

examples = [
    '降息了，该买点什么',  # this is the same sentence tried earlier
    '最近股票涨得挺好，什么原因？',
    '股市跌到底了吗',
    'ETF和股票基金有啥区别',
    'ETF和股票指数为啥不一样',
    'ETF基金有几种分别是啥',
    'ETF最少能买卖多少',
    'LOF与ETF有啥区别',
    'MOM基金与FOF基金相比有什么优势',
    'QDII基金与国内开放式基金哪个更好',
]


def print_my_examples(inputs, results):
    result_for_printing = [
        'input: {:30} : class: {}'.format(
            inputs[j], class_names[tf.argmax(results[j])])
        for j in range(len(inputs))]
    print(*result_for_printing, sep='\n')
    print()


# reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))
original_results = tf.sigmoid(classifier_model(tf.constant(examples)))

# print('Results from the saved model:')
# print_my_examples(examples, reloaded_results)
print('Results from the model in memory:')
print_my_examples(examples, original_results)


# Inference of queries of large scales from an excel file

print('Inference of queries of large scales:')

root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_dir = os.path.join(root_dir, 'BetaData')
data_file_str = os.path.join(data_root_dir, 'mydb.combined.xlsx')
df = pd.read_excel(
    data_file_str, header=1, names=['text'], engine='openpyxl')

original_results = tf.sigmoid(
    classifier_model.predict(tf.constant(df.text), batch_size=batch_size))
df['label'] = [class_names[tf.argmax(original_results[j])]
               for j in range(len(original_results))]
print(df)

data_file_str = os.path.join(data_root_dir, 'mydb.prediction.xlsx')
df.to_excel(data_file_str, header=1, index=False, engine='openpyxl')
print(f'prediction exported to {data_file_str}')
print()


# Inference of a single query for demo purpose

print('Inference of a query:')


def predict_class_of_a_query(query):
    result = tf.sigmoid(classifier_model(tf.constant(query)))
    print('input: {:30} : class: {}'.format(
        query[0], class_names[tf.argmax(result[0])]))
    print()


input_query = ['中欧蓝筹的申购费率是多少']
predict_class_of_a_query(input_query)


# Inference from console input

def predict_class_from_console_inputs():
    print('Inference from console inputs:')
    print('how many inputs (a positive integer): ', end='')
    size = int(input())
    for j in range(size):
        print('input {}: '.format(j), end='')
        query = [input()]
        result = tf.sigmoid(classifier_model(tf.constant(query)))
        print('class: {}'.format(class_names[tf.argmax(result[0])]))


predict_class_from_console_inputs()
