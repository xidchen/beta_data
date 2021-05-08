import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Load intent classes
saved_class_path = './intent_classes.txt'
with open(saved_class_path, encoding='utf-8') as f:
    intent_classes = f.read().strip().split('\n')

# Load saved model
saved_model_path = './beta_bert_intent_238'
saved_model_config = 'bert_zh_L-12_H-768_A-12/3'
intent_model = tf.saved_model.load(saved_model_path)
print(f'Model configuration: {saved_model_config}')

# Inference of queries of large scale from an excel file
print('Inference of queries of large scale:')

root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_dir = os.path.join(root_dir, 'BetaData')
data_file_str = os.path.join(data_root_dir, 'colleague_intent_input.xlsx')
df = pd.read_excel(data_file_str, header=0, names=['text'], engine='openpyxl')
print(df)

intent_indices = tf.argsort(
    [tf.sigmoid(intent_model([tf.constant(t)]))[0] for t in df.text], axis=1,
    direction='DESCENDING')
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

# Inference from console input
print('How many inputs (a positive integer): ', end='')
try:
    size = int(input())
except ValueError:
    size = 0

for i in range(size):
    print(f'Input {i}: ', end='')
    result = tf.sigmoid(model(tf.constant([input()])))
    print(f'Class: {class_names[tf.argmax(result[0])]}')

exit(0)
