# Setup

import abc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf

pd.set_option('expand_frame_repr', False)
for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


# The weather dataset

root_dir = os.path.dirname(os.path.realpath(__file__))
zip_path = tf.keras.utils.get_file(
    fname='jena_climate_2009_2016.csv.zip',
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
           'jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir=os.path.join(root_dir, 'BetaData', 'time')
)
df = pd.read_csv(os.path.splitext(zip_path)[0])

df = df[5::6]
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
plot_features = df[plot_cols]
plot_features.index = date_time
plot_features.plot(subplots=True)
plt.show()

plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
plot_features.plot(subplots=True)
plt.show()

# Inspect and cleanup
print(df.describe().transpose())
df.loc[df['wv (m/s)'] < 0, 'wv (m/s)'] = 0
df.loc[df['max. wv (m/s)'] < 0, 'max. wv (m/s)'] = 0
print(df['wv (m/s)'].min(), df['max. wv (m/s)'].min())

# Feature engineering
plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=50, vmax=400)
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')
plt.show()

wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')
wd_rad = df.pop('wd (deg)') * np.pi / 180
df['Wx'] = wv * np.cos(wd_rad)
df['Wy'] = wv * np.sin(wd_rad)
df['max Wx'] = max_wv * np.cos(wd_rad)
df['max Wy'] = max_wv * np.sin(wd_rad)

plt.hist2d(df['Wx'], df['Wy'], bins=50, vmax=400)
plt.colorbar()
plt.xlabel('Wind X [m/s]')
plt.ylabel('Wind Y [m/s]')
plt.show()

timestamp_s = date_time.map(pd.Timestamp.timestamp)
days_in_year = 365.2425
seconds_in_day = 86400
seconds_in_year = seconds_in_day * days_in_year
df['Day sin'] = np.sin(timestamp_s * 2 * np.pi / seconds_in_day)
df['Day cos'] = np.cos(timestamp_s * 2 * np.pi / seconds_in_day)
df['Year sin'] = np.sin(timestamp_s * 2 * np.pi / seconds_in_year)
df['Year cos'] = np.cos(timestamp_s * 2 * np.pi / seconds_in_year)

plt.plot(np.array(df['Day sin'])[:120])
plt.plot(np.array(df['Day cos'])[:120])
plt.xlabel('Time [h]')
plt.title('Time of day signal')
plt.show()

for col in df.columns[:2]:
    fft = tf.signal.rfft(df[col])
    f_per_dataset = np.arange(0, len(fft))

    hours_in_day = 24
    hours_in_year = hours_in_day * days_in_year
    years_per_dataset = len(df) / hours_in_year

    f_per_year = f_per_dataset / years_per_dataset
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.ylim(0, 400000)
    plt.xlim(0.1, max(plt.xlim()))
    plt.xticks([1, days_in_year], labels=['1/year', '1/day'])
    plt.xlabel(f'{col}: Frequency (log scale)')
    plt.title('Fast Fourier Transform')
    plt.show()

# Split the data
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[:int(n * .7)]
val_df = df[int(n * .7):int(n * .9)]
test_df = df[int(n * .9):]

num_features = df.shape[1]

# Normalize the data
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (df - train_mean) / train_std
df_std = pd.melt(df_std, var_name='Column', value_name='Normalized')
plt.figure(figsize=(20, 16))
plt.title('Distribution of features')
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
ax.set_xticks(np.arange(len(df.columns)))
ax.set_xticklabels(df.keys(), fontdict={'fontweight': 'bold'}, rotation=90)
plt.show()


# Data windowing

print('Data windowing')
print()


# 1. Indexes and offsets
class WindowGenerator:
    def __init__(self, _input_width: int, _label_width: int, _shift: int,
                 _train_df=train_df, _val_df=val_df, _test_df=test_df,
                 _label_columns=None):
        """Include all necessary logic for the input and label indices,
        and take the training, evaluation, and test DataFrames as input
        """
        # Store the raw data.
        self.train_df = _train_df
        self.val_df = _val_df
        self.test_df = _test_df

        # Work out the label column indices.
        self.label_columns = _label_columns
        if _label_columns:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(_label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = _input_width
        self.label_width = _label_width
        self.shift = _shift

        self.total_window_size = _input_width + _shift

        self.input_slice = slice(0, _input_width)
        self.input_indices = np.arange(_input_width + _shift)[self.input_slice]

        self.label_start = _input_width + _shift - _label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(_input_width + _shift)[self.label_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total Window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


w1 = WindowGenerator(_input_width=24, _label_width=1, _shift=24,
                     _label_columns=['T (degC)'])
print(w1)
w2 = WindowGenerator(_input_width=6, _label_width=1, _shift=1,
                     _label_columns=['T (degC)'])
print(w2)


# 2. Split
def split_window(self, _features: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """Split windows of features to a window of inputs and a window of labels
    :param self:
    :param _features: a list of consecutive inputs
    :return: inputs and labels pair
    """
    inputs = _features[:, self.input_slice, :]
    labels = _features[:, self.label_slice, :]
    if self.label_columns:
        labels = tf.stack([labels[:, :, self.column_indices[name]]
                           for name in self.label_columns], axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


WindowGenerator.split_window = split_window

# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100 + w2.total_window_size]),
                           np.array(train_df[200:200 + w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')


# 3. Plot
w2.example = example_inputs, example_labels


def plot(self,
         _model=None,
         _title=None,
         _plot_col='T (degC)',
         _max_subplots=3):
    """Visualize the split window"""
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[_plot_col]
    max_n = min(_max_subplots, len(inputs))
    for _n in range(max_n):
        plt.subplot(max_n, 1, _n + 1)
        if _n == 0:
            plt.title(_title)
        plt.ylabel(f'{_plot_col} [normed]')
        plt.plot(self.input_indices, inputs[_n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(_plot_col, None)
        else:
            label_col_index = plot_col_index

        plt.scatter(self.label_indices, labels[_n, :, label_col_index],
                    s=64, c='#2ca02c', edgecolors='k', label='Labels')
        if _model is not None:
            predictions = _model(inputs)
            plt.scatter(self.label_indices,
                        predictions[_n, :, label_col_index],
                        s=64, c='#ff7f0e', marker='X', edgecolors='k',
                        label='Predictions')
        if _n == 0:
            plt.legend()

    plt.xlabel('Time [h]')


WindowGenerator.plot = plot

w2.plot()
plt.show()

w2.plot(_plot_col='p (mbar)')
plt.show()


# 4. Create tf.data.Datasets
def make_dataset(self, _data: pd.DataFrame) -> tf.data.Dataset:
    """Create a function to take a time series DataFrame and
    convert it to a tf.data.Dataset of (input_window, label_window) pairs
    :param self:
    :param _data: time series DataFrame
    :return: (input_window, label_window) pairs
    """
    _data = np.array(_data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=_data,
        targets=None,
        sequence_length=self.total_window_size,
        batch_size=32,
        shuffle=True)
    ds = ds.map(self.split_window)
    return ds


WindowGenerator.make_dataset = make_dataset


@property
def train(self):
    return self.make_dataset(self.train_df)


@property
def val(self):
    return self.make_dataset(self.val_df)


@property
def test(self):
    return self.make_dataset(self.test_df)


@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting"""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset.
        result = next(iter(self.train))
        # And cache it for next time.
        self._example = result
    return result


WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

# Each element is an (inputs, label) pair.
print(w2.train.element_spec)

# Iterating over a Dataset yields concrete batches.
for example_inputs, example_labels in w2.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
print()


# Single step models
"""
The simplest model on this sort of data is one that predicts 
a single feature's value -- 1 time step (one hour) into the future 
based only on the current conditions. 
"""

print('Single step models')
print()
single_step_window = WindowGenerator(
    _input_width=1, _label_width=1, _shift=1, _label_columns=['T (degC)'])
print(single_step_window)

single_step_window.plot(_title='Single step window')
plt.show()

for example_inputs, example_labels in single_step_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
print()


# Baseline
class Baseline(tf.keras.Model, abc.ABC):
    def __init__(self, label_index=None):
        super(Baseline, self).__init__()
        self.label_index = label_index

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        if self.label_index is None:
            return inputs
        results = inputs[:, :, self.label_index]
        return results[:, :, tf.newaxis]


print('Baseline')
baseline = Baseline(label_index=column_indices['T (degC)'])
baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance, performance = {}, {}
print('Validation set performance:')
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
print('Test set performance:')
performance['Baseline'] = baseline.evaluate(single_step_window.test)

wide_window = WindowGenerator(
    _input_width=24, _label_width=24, _shift=1, _label_columns=['T (degC)'])
print(wide_window)

print('Wide window')
print(f'Input shape: {wide_window.example[0].shape}')
print(f'Output shape: {baseline(wide_window.example[0]).shape}')

wide_window.plot(baseline, 'Baseline')
plt.show()
print()


# Linear model
print('Linear model')
linear = tf.keras.Sequential([tf.keras.layers.Dense(1)])

print(f'Input shape: {single_step_window.example[0].shape}')
print(f'Output shape: {linear(single_step_window.example[0]).shape}')

MAX_EPOCHS = 20


def compile_and_fit(_model: tf.keras.Model,
                    _window: WindowGenerator,
                    _patience: int = 2) -> tf.keras.callbacks.History:
    """Package the training procedure into a function
    :param _model: a tf.keras.Model object
    :param _window: a WindowGenerator object
    :param _patience: number of epochs with no improvement for early stopping
    :return: a History object
    """
    _early_stopping = tf.keras.callbacks.EarlyStopping(patience=_patience)
    _model.compile(optimizer=tf.optimizers.Adam(),
                   loss=tf.losses.MeanSquaredError(),
                   metrics=[tf.metrics.MeanAbsoluteError()])
    _history = _model.fit(_window.train, epochs=MAX_EPOCHS,
                          callbacks=[_early_stopping],
                          validation_data=_window.val)
    return _history


history_linear = compile_and_fit(linear, single_step_window)

print('Validation set performance:')
val_performance['Linear'] = linear.evaluate(single_step_window.val)
print('Test set performance:')
performance['Linear'] = linear.evaluate(single_step_window.test)

print('Wide window')
print(f'Input shape: {wide_window.example[0].shape}')
print(f'Output shape: {linear(wide_window.example[0]).shape}')

wide_window.plot(linear, 'Linear model')
plt.show()

plt.figure(figsize=(18, 14))
plt.title('Weights assigned to each input')
plt.bar(x=range(len(train_df.columns)),
        height=linear.layers[0].kernel[:, 0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
axis.set_xticklabels(train_df.columns, rotation=20)
plt.show()
print()


# Dense
print('Dense')
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

history_dense = compile_and_fit(dense, single_step_window)

print('Validation set performance:')
val_performance['Dense'] = dense.evaluate(single_step_window.val)
print('Test set performance:')
performance['Dense'] = dense.evaluate(single_step_window.test)

print('Wide window')
print(f'Input shape: {wide_window.example[0].shape}')
print(f'Output shape: {dense(wide_window.example[0]).shape}')

wide_window.plot(dense, 'Dense')
plt.show()
print()


# Multi-step dense
print('Multi-step dense')
CONV_WIDTH = 3
conv_window = WindowGenerator(
    _input_width=CONV_WIDTH, _label_width=1, _shift=1,
    _label_columns=['T (degC)'])
print(conv_window)

conv_window.plot(_title='Conv window')
plt.show()

multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time * features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1),
    # Add back the time dimension.
    # Shape: (outputs) => ([1, outputs])
    tf.keras.layers.Reshape((1, -1))
])

print(f'Input shape: {conv_window.example[0].shape}')
print(f'Output shape: {multi_step_dense(conv_window.example[0]).shape}')

history_multi_step_dense = compile_and_fit(multi_step_dense, conv_window)

print('Validation set performance:')
val_performance['Multi-step dense'] = multi_step_dense.evaluate(conv_window.val)
print('Test set performance:')
performance['Multi-step dense'] = multi_step_dense.evaluate(conv_window.test)

# The main downside of this approach is that the resulting model can only be
# executed on input windows of exactly this shape.
print('Wide window')
print(f'Input shape: {wide_window.example[0].shape}')
try:
    print(f'Output shape: {multi_step_dense(wide_window.example[0]).shape}')
except Exception as e:
    print(f'{type(e).__name__}: {e}')
# The convolutional models in the next section fix this problem.
print()


# CNN
print('Conv')
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, CONV_WIDTH, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

print(f'Input shape: {conv_window.example[0].shape}')
print(f'Output shape: {conv_model(conv_window.example[0]).shape}')

history_conv = compile_and_fit(conv_model, conv_window)

print('Validation set performance:')
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
print('Test set performance:')
performance['Conv'] = conv_model.evaluate(conv_window.test)

print('Wide window')
print(f'Input shape: {wide_window.example[0].shape}')
print(f'Label shape: {wide_window.example[1].shape}')
print(f'Output shape: {conv_model(wide_window.example[0]).shape}')

LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + CONV_WIDTH - 1
wide_conv_window = WindowGenerator(
    _input_width=INPUT_WIDTH, _label_width=LABEL_WIDTH, _shift=1,
    _label_columns=['T (degC)'])
print(wide_conv_window)

print('Wide conv window')
print(f'Input shape: {wide_conv_window.example[0].shape}')
print(f'Label shape: {wide_conv_window.example[1].shape}')
print(f'Output shape: {conv_model(wide_conv_window.example[0]).shape}')

wide_conv_window.plot(conv_model, 'Conv')
plt.show()
print()


# RNN
print('LSTM')
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(1)
])

print(f'Input shape: {wide_window.example[0].shape}')
print(f'Output shape: {lstm_model(wide_window.example[0]).shape}')

history_lstm = compile_and_fit(lstm_model, wide_window)

print('Validation set performance:')
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
print('Test set performance:')
performance['LSTM'] = lstm_model.evaluate(wide_window.test)

print('Wide window')
print(f'Input shape: {wide_window.example[0].shape}')
print(f'Label shape: {wide_window.example[1].shape}')
print(f'Output shape: {lstm_model(wide_window.example[0]).shape}')

wide_window.plot(lstm_model, 'LSTM')
plt.show()
print()


# Performance
print('Performance')

x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index(metric_name)
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel(f'{metric_name} [T (degC), normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(x, labels=performance.keys(), rotation=10)
plt.title('Performance')
plt.legend()
plt.show()

for name, value in performance.items():
    print(f'{name:16s}: {value[1]:.4f}')
print()


# Multi-output models
print('Multi-output models')

single_step_window = WindowGenerator(_input_width=1, _label_width=1, _shift=1)

wide_window = WindowGenerator(_input_width=24, _label_width=24, _shift=1)

print('Wide window')
for example_inputs, example_labels in wide_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
print()


# Baseline
print('Baseline')
baseline = Baseline()
baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance, performance = {}, {}
print('Validation set performance:')
val_performance['Baseline'] = baseline.evaluate(wide_window.val)
print('Test set performance:')
performance['Baseline'] = baseline.evaluate(wide_window.test)
print()


# Dense
print('Dense')
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_features)
])

history_dense = compile_and_fit(dense, single_step_window)

print('Validation set performance:')
val_performance['Dense'] = dense.evaluate(single_step_window.val)
print('Test set performance:')
performance['Dense'] = dense.evaluate(single_step_window.test)
print()


# RNN
print('LSTM')
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(num_features)
])

history_lstm = compile_and_fit(lstm_model, wide_window)

print('Validation set performance:')
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
print('Test set performance:')
performance['LSTM'] = lstm_model.evaluate(wide_window.test)
print()


# Advanced: Residual connections
class ResidualWrapper(tf.keras.Model, abc.ABC):
    def __init__(self, _model: tf.keras.Model):
        super(ResidualWrapper, self).__init__()
        self.model = _model

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        delta = self.model(inputs, *args, **kwargs)
        # The prediction for each time step is the input from the previous
        # time step plus the delta calcuated by the model.
        return inputs + delta


print('Residual LSTM')
residual_lstm = ResidualWrapper(
    tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(
            num_features,
            # The predicted deltas should start small.
            # Therefore, initialize the output layer with zeros.
            kernel_initializer=tf.initializers.zeros())
    ]))

history_residual_lstm = compile_and_fit(residual_lstm, wide_window)

print('Validation set performance:')
val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val)
print('Test set performance:')
performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test)
print()


# Performance
print('Performance')

x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index(metric_name)
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel(f'{metric_name} (average over all outputs)')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(x, labels=performance.keys(), rotation=10)
plt.title('Performance')
plt.legend()
plt.show()

for name, value in performance.items():
    print(f'{name:16s}: {value[1]:.4f}')
print()


# Multi-step models
"""
Both the single-output and multiple-output models in the previous 
sections made single time step predictions, one hour into the future.
This section looks at how to expand these models to make 
multiple time step predictions.
There are two rough approaches to this:
    1. Single shot predictions where the entire time series is predicted 
    at once.
    2. Autoregressive predictions where the model only makes single step 
    predictions and its output is fed back as its input.
In this section all the models will predict all the features 
across all output time steps.
"""

print('Multi-step models')
print()

OUT_STEPS = 24
multi_window = WindowGenerator(
    _input_width=24, _label_width=OUT_STEPS, _shift=OUT_STEPS)
print(multi_window)

multi_window.plot(_title='Multi-step window')
plt.show()

for example_inputs, example_labels in multi_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Lables shape (batch, time, features): {example_labels.shape}')
print()


# Baseline
class MultiStepLastBaseline(tf.keras.Model, abc.ABC):
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Repeat the last input time step for the required number
        of output time steps"""
        return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])


print('Multi-step Last Baseline')
last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])

multi_val_performance = {}
multi_performance = {}
print('Validation set performance:')
multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
print('Test set performance:')
multi_performance['Last'] = last_baseline.evaluate(multi_window.test)
print()

multi_window.plot(last_baseline, 'Last Baseline')
plt.show()


class RepeatBaseline(tf.keras.Model, abc.ABC):
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Repeat the previous day, assuming tomorrow will be similar"""
        return inputs


print('Multi-step Repeat Baseline')
repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])

print('Validation set performance:')
multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
print('Test set performance:')
multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test)
print()

multi_window.plot(repeat_baseline, 'Repeat Baseline')
plt.show()


# Single-shot models
print('Single-shot models')
print()


# Linear
print('Multi-step Linear')
multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda _x: _x[:, -1:, :]),
    # Shape => [batch, 1, out_steps * features]
    tf.keras.layers.Dense(OUT_STEPS * num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape((OUT_STEPS, num_features))
])

history_multi_linear = compile_and_fit(multi_linear_model, multi_window)

print('Validation set performance:')
multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
print('Test set performance:')
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test)
print()

multi_window.plot(multi_linear_model, 'Linear')
plt.show()


# Dense
print('Multi-step Dense')
multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda _x: _x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps * features]
    tf.keras.layers.Dense(OUT_STEPS * num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape((OUT_STEPS, num_features))
])

history_multi_dense = compile_and_fit(multi_dense_model, multi_window)

print('Validation set performance:')
multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
print('Test set performance:')
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test)
print()

multi_window.plot(multi_dense_model, 'Dense')
plt.show()


# CNN
print('Multi-step Conv')
CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda _x: _x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, kernel_size=CONV_WIDTH, activation='relu'),
    # Shape => [batch, 1, out_steps * features]
    tf.keras.layers.Dense(OUT_STEPS * num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape((OUT_STEPS, num_features))
])

history_multi_conv = compile_and_fit(multi_conv_model, multi_window)

print('Validation set performance:')
multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
print('Test set performance:')
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test)
print()

multi_window.plot(multi_conv_model, 'Conv')
plt.show()


# RNN
print('Multi-step LSTM')
multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps * features]
    tf.keras.layers.Dense(OUT_STEPS * num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape((OUT_STEPS, num_features))
])

history_multi_lstm = compile_and_fit(multi_lstm_model, multi_window)

print('Validation set performance:')
multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
print('Test set performance:')
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test)
print()

multi_window.plot(multi_lstm_model, 'LSTM')
plt.show()


# Advanced: Autoregressive model
"""
The above models all predict the entire output sequence in a single step.
It may be helpful to decompose this prediction into individual time steps.
Then, each model's output can be fed back into itself at each step and
predictions can be made conditioned on the previous one, like in the classic 
Generating Sequences With RNNs (https://arxiv.org/abs/1308.0850). 
"""
print('Autoregressive model')


# RNN
class FeedBack(tf.keras.Model, abc.ABC):
    def __init__(self, units: int, out_steps: int):
        super(FeedBack, self).__init__()
        self.units = units
        self.out_steps = out_steps
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)


print('Autoregressive RNN')
feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)


def warmup(self, inputs: tf.Tensor) -> (tf.Tensor, [tf.Tensor, tf.Tensor]):
    """Initialize model internal state based on the inputs. Once trained,
    this state will capture the relevant parts of the input history."""
    # inputs.shape => (batch, time, features)
    # x.shape => (batch, lstm_units)
    _x, *_state = self.lstm_rnn(inputs)
    # predictions.shape => (batch, features)
    _prediction = self.dense(_x)
    return _prediction, _state


FeedBack.warmup = warmup

prediction, state = feedback_model.warmup(multi_window.example[0])
print(f'Prediction shape: {prediction.shape}')


def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
    """Iterate the model feeding the predictions at each step back
    as the input"""
    # Use a TensorArray to capture dynamically unrolled outputs.
    predictions = []
    # Initialize the LSTM state.
    _prediction, _state = self.warmup(inputs)
    # Insert the first prediction.
    predictions.append(_prediction)
    # Run the rest of the prediction steps.
    for _n in range(1, self.out_steps):
        # Use the last prediction as inputs.
        _x = _prediction
        # Execute one lstm step.
        _x, _state = self.lstm_cell(_x, states=_state, training=training)
        # Convert the lstm output to a prediction.
        _prediction = self.dense(_x)
        # Add the prediction to the output.
        predictions.append(_prediction)
    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])
    return predictions


FeedBack.call = call

print(f'Input shape: {multi_window.example[0].shape}')
print(f'Output shape: {feedback_model(multi_window.example[0]).shape}')

history_autoregressive_lstm = compile_and_fit(feedback_model, multi_window)

print('Validation set performance:')
multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
print('Test set performance:')
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test)
print()

multi_window.plot(feedback_model, 'AR LSTM')
plt.show()


# Performance
print('Performance')

x = np.arange(len(multi_performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = multi_lstm_model.metrics_names.index(metric_name)
multi_val_mae = [v[metric_index] for v in multi_val_performance.values()]
multi_test_mae = [v[metric_index] for v in multi_performance.values()]

plt.ylabel(f'{metric_name} (average over all times and outputs)')
plt.bar(x - 0.17, multi_val_mae, width, label='Validation')
plt.bar(x + 0.17, multi_test_mae, width, label='Test')
plt.xticks(x, labels=multi_performance.keys(), rotation=10)
plt.legend()
plt.show()

for name, value in multi_performance.items():
    print(f'{name:16s}: {value[1]:.4f}')
print()
