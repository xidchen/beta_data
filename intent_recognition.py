import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text


# Load intent classes
saved_class_path = './intent_classes.txt'
with open(saved_class_path, encoding='utf-8') as f:
    class_names = f.read().strip().split('\n')


# Load saved model
saved_model_path = './beta_bert_intent_229'
saved_model_config = 'bert_zh_L-12_H-768_A-12/3'
model = tf.saved_model.load(saved_model_path)
print(f'Model configuration: {saved_model_config}')


# Inference
print('How many inputs (a positive integer): ', end='')
try:
    size = int(input())
except ValueError:
    size = 0

for i in range(size):
    print(f'Input {i}: ', end='')
    result = tf.sigmoid(model(tf.constant([input()])))
    print(f'Class: {class_names[tf.argmax(result[0])]}')
