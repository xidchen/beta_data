# Setup

import glob
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import tensorflow as tf

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


# Import the mini Speech Commands dataset

root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_dir = os.path.join(root_dir, 'BetaData', 'speech')
if not os.path.exists(data_root_dir):
    os.makedirs(data_root_dir)
zip_path = tf.keras.utils.get_file(
    fname='mini_speech_commands.zip',
    origin='https://storage.googleapis.com/download.tensorflow.org/data/'
           'mini_speech_commands.zip',
    extract=True,
    cache_dir=data_root_dir
)
data_dir = os.path.join(data_root_dir, 'datasets', 'mini_speech_commands')

commands = np.array(os.listdir(data_dir))
commands = commands[commands != 'README.md']
print(f'Commands: {commands}')

filenames = glob.glob(data_dir + '/*/*')
filenames = tf.random.shuffle(filenames)
print(f'Number of total examples: {len(filenames)}')
print(f'Number of examples per label: '
      f'{len(os.listdir(os.path.join(data_dir, commands[0])))}')
print(f'Example file tensor: {filenames[0]}')

train_files = filenames[:6400]
val_files = filenames[6400:6400 + 800]
test_files = filenames[-800:]
print(f'Training set size: {len(train_files)}')
print(f'Validation set size: {len(val_files)}')
print(f'Test set size: {len(test_files)}')


# Read the audio files and their labels

def decode_audio(audio_binary: tf.Tensor) -> tf.Tensor:
    """Decode WAV-encoded audio files to `float32` tensors, normalized
    to the [-1.0, 1.0] range. Return `float32` audio and a sample rate
    :param audio_binary: WAV-encoded audio binary tensor
    :return: `float32`-decoded audio tensor
    """
    _audio, _ = tf.audio.decode_wav(audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(_audio, axis=-1)


def get_label(file_path: tf.Tensor) -> tf.Tensor:
    """Define a function that creates labels using the parent directories
    for each file
    :param file_path: WAV audio file path tensor
    :return: audio file label tensor
    """
    _parts = tf.strings.split(input=file_path, sep=os.path.sep)
    return _parts[-2]


def get_waveform_and_label(file_path: tf.Tensor) -> tuple:
    """Define a function that puts waveform and label all together
    :param file_path: WAV audio file path tensor
    :return: a tuple containing the audio and label tensors
    """
    _label = get_label(file_path)
    _audio_binary = tf.io.read_file(file_path)
    _waveform = decode_audio(_audio_binary)
    return _waveform, _label


AUTOTUNE = tf.data.AUTOTUNE

files_ds = tf.data.Dataset.from_tensor_slices(train_files)

waveform_ds = files_ds.map(
    map_func=get_waveform_and_label,
    num_parallel_calls=AUTOTUNE)

nrows, ncols = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))

for i, (audio, label) in enumerate(waveform_ds.take(nrows * ncols)):
    r, c = i // ncols, i % ncols
    ax = axes[r][c]
    ax.plot(audio.numpy())
    ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
    label = label.numpy().decode('utf-8')
    ax.set_title(label)

plt.show()


# Convert waveforms to spectrograms

def get_spectrogram(_waveform: tf.Tensor) -> tf.Tensor:
    """Create a utility function for converting waveforms to spectrograms
    :param _waveform: time-domain signals
    :return: time-frequency-domain signals
    """
    _input_len = 16000
    # Zero-padding for an audio waveform with less than 16,000 samples.
    _zero_padding = tf.zeros([_input_len] - tf.shape(_waveform[:_input_len]))
    # Ensure all audio clips are of the same length.
    _waveform = tf.concat([_waveform, _zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    _spectrogram = tf.signal.stft(_waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    _spectrogram = tf.abs(_spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers, which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    _spectrogram = _spectrogram[..., tf.newaxis]
    return _spectrogram


def plot_spectrogram(_spectrogram: tf.Tensor, _ax: matplotlib.axes.Axes):
    """Define a function for displaying a spectrogram
    :param _spectrogram: time-frequency-domain signals
    :param _ax: a subplot for displaying
    """
    if len(_spectrogram.shape) > 2:
        assert len(_spectrogram.shape) == 3
        _spectrogram = np.squeeze(_spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    _log_spec = np.log(_spectrogram.T + np.finfo(float).eps)
    _height = _log_spec.shape[0]
    _width = _log_spec.shape[1]
    _x = np.linspace(0, np.size(_spectrogram), num=_width, dtype=int)
    _y = range(_height)
    _ax.pcolormesh(_x, _y, _log_spec, shading='auto')


for waveform, label in waveform_ds.take(1):
    label = label.numpy().decode('utf-8')
    spectrogram = get_spectrogram(waveform)
    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform.numpy())
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 16000])
    plot_spectrogram(spectrogram, axes[1])
    axes[1].set_title('Spectrogram')
    plt.show()


def get_spectrogram_and_label_id(_audio: tf.Tensor, _label: tf.Tensor) -> tuple:
    """Define a function that transforms the waveform dataset into spectrograms
    and their corresponding labels as integer IDs
    :param _audio: the waveform dataset
    :param _label: the waveform label
    :return: spectrogram and label integer ID
    """
    _spectrogram = get_spectrogram(_audio)
    _label_id = tf.argmax(_label == commands)
    return _spectrogram, _label_id


spectrogram_ds = waveform_ds.map(
    map_func=get_spectrogram_and_label_id,
    num_parallel_calls=AUTOTUNE)

nrows, ncols = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))

for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(nrows * ncols)):
    r, c = i // ncols, i % ncols
    ax = axes[r][c]
    plot_spectrogram(spectrogram, ax)
    ax.set_title(commands[label_id.numpy()])

plt.show()


# Build and train the model

def preprocess_dataset(_files: tf.Tensor) -> tf.data.Dataset:
    """Repeat the training set preprocessing on the validation and test sets
    :param _files: the validation or the test audio files
    :return: the validation or the test datasets
    """
    _files_ds = tf.data.Dataset.from_tensor_slices(_files)
    _waveform_ds = _files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE)
    _spectrogram_ds = _waveform_ds.map(
        map_func=get_spectrogram_and_label_id,
        num_parallel_calls=AUTOTUNE)
    return _spectrogram_ds


train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

"""
The `tf.keras.Sequential` model uses the following Keras preprocessing layers.
    * `tf.keras.layers.experimental.preprocessing.Resizing`:
        to downsample the input to enable the model to train faster.
    * `tf.keras.layers.experimental.preprocessing.Normalization`:
        to normalize each pixel in the image based on its mean and std dev.
For the `Normalization` layer, its `adapt` method would first need to be called
on the training data in order to compute aggregate statistics (mean and std dev)
"""

input_shape = None
for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
print(f'Input shape: {input_shape}')
num_labels = len(commands)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
# Fit the state of the layer to the spectrograms with `Normalization.adapt`
norm_layer.adapt(spectrogram_ds.map(map_func=lambda _s, _l: _s))

model = tf.keras.Sequential([
    tf.keras.Input(shape=input_shape),
    tf.keras.layers.experimental.preprocessing.Resizing(32, 32),
    norm_layer,
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_labels)
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(patience=2, verbose=1)
)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()


# Evaluate the model performance

test_audio = []
test_labels = []

for audio, label in test_ds:
    test_audio.append(audio.numpy())
    test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g',
            xticklabels=commands, yticklabels=commands)
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()


# Run inference on an audio file

sample_file = os.path.join(data_dir, 'no', '01bb6a2a_nohash_0.wav')

sample_ds = preprocess_dataset(tf.constant([sample_file]))

for spectrogram, label in sample_ds.batch(1):
    prediction = model(spectrogram)
    plt.bar(commands, tf.nn.softmax(prediction[0]))
    plt.title(f'Predictions for "{commands[label[0]]}"')
    plt.show()
