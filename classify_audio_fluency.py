import numpy as np
import os
import pandas as pd
import sklearn.model_selection as sms
import tensorflow as tf

import beta_audio

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_dir = os.path.join(root_dir, 'BetaData', 'coach', 'ml', 'fluency')


# Feature extraction

audio_subdirectories = os.listdir(data_root_dir)
audio_subdirectories = sorted(audio_subdirectories)
print(f'Audio subdirs: {audio_subdirectories}')

features, labels, files = beta_audio.parse_audio_files(
    data_root_dir, audio_subdirectories)
print()

# Data exploration

label_classes = np.array(audio_subdirectories)
n_features = features.shape[-1]
n_classes = len(label_classes)


# Train, valid, test split
print('Data splitting')

print(f'Dataset total length:  {len(labels)}')
x_raw, x_test, y_raw, y_test, f_raw, f_test = sms.train_test_split(
    features, labels, files, test_size=0.1, random_state=0, stratify=labels)
x_train, x_val, y_train, y_val, f_train, f_val = sms.train_test_split(
    x_raw, y_raw, f_raw, train_size=0.85, random_state=0, stratify=y_raw)
print(f'Training set length:   {len(y_train)}')
print(f'Validation set length: {len(y_val)}')
print(f'Test set length:       {len(y_test)}')
print()


# MLP
print('MLP')

mlp = tf.keras.Sequential([
    tf.keras.Input(shape=(n_features,)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)
], name='mlp')

mlp.summary()

mlp.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callback_0 = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=20, verbose=1, restore_best_weights=True)
callback_1 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

print('Training:')
history_mlp = mlp.fit(
    x_train, y_train, epochs=100, batch_size=128,
    callbacks=[callback_0, callback_1], validation_data=(x_val, y_val),
)
print()

print('Evaluation:')
mlp.evaluate(x_test, y_test, batch_size=1)
y_pred = np.argmax(mlp.predict(x_test), axis=1)
print()


print('Confusion matrix:')
print(pd.crosstab(label_classes[y_test], label_classes[y_pred],
                  rownames=['actual'], colnames=['predicted']))
print()


print('Prediction:')
for f in f_test:
    features = np.array(beta_audio.feature_extraction(
        f, dims=[20, 1, 1, 1], gate=[1, 0, 1, 1]), ndmin=2)
    prediction = mlp(features)
    grades = [float(lc) * 11 for lc in label_classes]
    score = int(tf.tensordot(prediction, grades, axes=1).numpy()[0])
    prediction = [round(p, 4) for p in prediction.numpy()[0]]
    predicted_label = [label_classes[np.argmax(prediction)]]
    print(f'{f.split(os.sep)[-2:]}: {score}, {predicted_label}, {prediction}')
print()