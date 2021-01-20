import pandas as pd
import tensorflow as tf
import tensorflow_text

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')
tf.constant(0)


# Load class name

batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'BetaData/train', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
class_names = raw_train_ds.class_names


# Load saved model

saved_model_path = './beta_bert'
print(f'Load saved models: {saved_model_path}')
loaded_model = tf.saved_model.load(saved_model_path)

examples = [
    '一帖看懂实操分级基金折价套利',
    '分级基金折价套利流程',
    '什么是开放式基金份额的转换，有哪些要注意的事项'
]
df = pd.DataFrame({'input': examples})
print('Inference of examples:')
print(df)
loaded_results = tf.sigmoid(loaded_model(tf.constant(examples)))


def print_my_examples(inputs, results):
    result_for_printing = [
        'input: {}\nclass: {}\n'.format(
            inputs[j],
            [(class_names[k], results[j][k].numpy().round(3))
             for k in tf.argsort(results[j], direction='DESCENDING')[:3]]
        ) for j in range(len(inputs))]
    print(*result_for_printing, sep='\n')
    print()


print('Results from the saved model:')
print_my_examples(examples, loaded_results)
