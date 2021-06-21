import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tensorflow.compat.v1 as tf

import clustering_data
import clustering_utils

np.set_printoptions(precision=2)
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_rows = 12
pd.set_option('max_columns', 12)
pd.set_option('display.max_colwidth', 12)
pd.set_option('expand_frame_repr', False)
tf.disable_v2_behavior()


# Load and clean data

print('Load and clean data')

choc_data = clustering_data.load_choc_data()
original_cols = choc_data.columns.values


# Process data

choc_data.drop(columns=['review_date', 'reference_number'], inplace=True)
print(choc_data)


# Generate embeddings from DNN


class SimilarityModel(object):
    """Class to build, train, and inspect a Similarity Model.

    This class builds a deep neural network that maps a dataset of entities
    with heterogenous features to an embedding space.
    Given a dataset as a pandas dataframe, determine the model by specifying
    the set of features used as input and as labels to the DNN, and the
    size of each hidden layer. The data is mapped to the embedding space
    in the last hidden layer.

    To build an auto-encoder, make the set of output features identical
    to the set of input features. Alternatively, build a predictor by using
    a single feature as the label. When using a single feature as a label,
    ensure this feature is removed from the input, or add at least one hidden
    layer of a sufficiently low dimension such that the model cannot trivially
    learn the label.
    Caveat: The total loss being minimized is a simple sum of losses for each
        output label (plus the regularization). If the output feature set
        combines sparse and dense features, the total loss is a sum of
        cross-entropy soft-max losses with root mean squared error losses,
        potentially in different scales, which could emphasis
        some output labels more than others.
    """

    def __init__(self, dataframe, _input_feature_names, _output_feature_names,
                 _dense_feature_names, _sparse_input_feature_embedding_dims,
                 _hidden_dims, _l2_regularization=0.0, use_bias=True,
                 batch_size=100, inspect=False):
        """Build a similarity model.

        Args:
          dataframe: the pandas dataframe used to train and validate the model.
          _input_feature_names: list of strings, names of input feature columns.
          _output_feature_names: list of strings, names of output feature columns.
          _dense_feature_names: list of strings, names of feature columns that are
            treated as dense. All other feature columns are treated as sparse.
          _sparse_input_feature_embedding_dims: dictionary that maps feature names
            to ints, expressing the embedding dimension of each input feature.
            Any sparse feature in input_feature_names must be in this dictionary.
          _hidden_dims: list of ints, dimensions of each hidden layer.
            These hidden layers are not counting the first layer which is
            a concatenation of the input embeddings and the dense input features.
            Hence, this list can be empty, in which case the outputs
            of the network are directly connected to the input embeddings
            and/or dense inputs.
          use_bias: bool, if true, add a bias term to each hidden layer.
          batch_size: int, batch size.
          inspect: bool, if true, add each tensor of the model to the list of
            tensors that are inspected.
        """
        used_feature_names = tuple(
            set(_input_feature_names).union(_output_feature_names))
        _sparse_feature_names = tuple(
            set(used_feature_names).difference(_dense_feature_names))
        # Dictionary mapping each sparse feature column to its vocabulary.
        # sparse_feature_vocabs = { 'maker': [u'A. Morin', u'AMMA', ...], ... }
        sparse_feature_vocabs = {sfn: sorted(list(set(choc_data[sfn].values)))
                                 for sfn in _sparse_feature_names}

        # Sparse output features are mapped to ids via tf.feature_to_id,
        # hence we need key-id pairs for these vocabularies.
        sparse_output_feature_names = (tuple(set(
            _sparse_feature_names).intersection(_output_feature_names)))
        keys_and_values = {}
        for fn in sparse_output_feature_names:
            keys = tf.constant(sparse_feature_vocabs[fn], dtype=tf.string,
                               name='{}_vocab_keys'.format(fn))
            values = tf.range(len(sparse_feature_vocabs[fn]), dtype=tf.int64,
                              name='{}_vocab_values'.format(fn))
            keys_and_values[fn] = (keys, values)

        # Class instance data members.
        self._session = None
        self._loss = None
        self._metrics = {}
        self._embeddings = None
        self._vars_to_inspect = {}

        def split_dataframe(df, holdout_fraction=0.1):
            """Splits a pandas dataframe into training and test sets.

            Args:
              df: the source pandas dataframe.
              holdout_fraction: fraction of dataframe rows to use in the test set.

            Returns:
              A pair of non-overlapping pandas dataframe for training and holdout.
            """
            test = df.sample(frac=holdout_fraction, replace=False)
            train = df[~df.index.isin(test.index)]
            return train, test

        train_dataframe, test_dataframe = split_dataframe(dataframe)

        def make_batch(_df, _batch_size):
            """Creates a batch of examples.

            Args:
              _df: a panda dataframe with rows being examples and with
                columns being feature columns.
              _batch_size: the batch size.

            Returns:
              A dictionary of tensors, keyed by their feature names.
              Each tensor is of shape [batch_size]. Tensors for sparse features
              are of strings, while tensors for dense features are of floats.
            """
            used_features = {ufn: _df[ufn] for ufn in used_feature_names}
            batch = (tf.data.Dataset.from_tensor_slices(used_features).shuffle(
                1000).repeat().batch(
                _batch_size).make_one_shot_iterator().get_next())
            if inspect:
                for _k, _v in batch.items():
                    self._vars_to_inspect['input_{}'.format(_k)] = _v
            return batch

        def generate_feature_columns(feature_names):
            """Creates the list of used feature columns.

            Args:
              feature_names: an iterable of strings with the names
                of the features for which feature columns are generated.

            Returns:
              A dictionary, keyed by feature names, of _DenseColumn and
              _NumericColumn.
            """
            used_sparse_feature_names = (
                tuple(set(_sparse_feature_names).intersection(feature_names)))
            used_dense_feature_names = (
                tuple(set(_dense_feature_names).intersection(feature_names)))
            f_columns = {}
            for sfn in used_sparse_feature_names:
                sf_column = tf.feature_column.categorical_column_with_vocabulary_list(
                    key=sfn, vocabulary_list=sparse_feature_vocabs[sfn],
                    num_oov_buckets=0)
                f_columns[sfn] = tf.feature_column.embedding_column(
                    categorical_column=sf_column,
                    dimension=_sparse_input_feature_embedding_dims[sfn],
                    combiner='mean',
                    initializer=tf.truncated_normal_initializer(stddev=.1))
            for dfn in used_dense_feature_names:
                f_columns[dfn] = tf.feature_column.numeric_column(dfn)
            return f_columns

        def create_tower(features, columns):
            """Creates the tower mapping features to embeddings.

            Args:
              features: a dictionary of tensors of shape [batch_size],
                keyed by feature name. Sparse features are associated
                to tensors of strings, while dense features are associated
                to tensors of floats.
              columns: a dictionary, keyed by feature names,
                of _DenseColumn and _NumericColumn.

            Returns:
              A pair of elements: hidden_layer and output_layer.
                hidden_layer is a tensor of shape [batch_size, hidden_dims[-1]].
                output_layer is a dictionary keyed by the output feature names,
                  of dictionaries {'labels': labels, 'logits': logits}.
                  Dense output features have both labels and logits
                  as float tensors of shape [batch_size, 1].
                  Sparse output features have labels as string tensors
                  of shape [batch_size, 1] and logits as float tensors
                  of shape [batch_size, len(sparse_feature_vocab)].
            """
            # Input features.
            input_columns = [columns[_fn] for _fn in _input_feature_names]
            hidden_layer = tf.feature_column.input_layer(
                features, input_columns)
            dense_input_feature_names = (tuple(
                set(_dense_feature_names).intersection(_input_feature_names)))
            input_dim = (sum(_sparse_input_feature_embedding_dims.values()) +
                         len(dense_input_feature_names))
            for layer_idx, layer_output_dim in enumerate(_hidden_dims):
                w = tf.get_variable(
                    'hidden{}_w_'.format(layer_idx),
                    shape=[input_dim, layer_output_dim],
                    initializer=tf.truncated_normal_initializer(
                        stddev=1.0 / np.sqrt(layer_output_dim)))
                if inspect:
                    self._vars_to_inspect['hidden{}_w_'.format(layer_idx)] = w
                hidden_layer = tf.matmul(hidden_layer, w)  # / 10.)
                if inspect:
                    self._vars_to_inspect[
                        'hidden_layer_{}'.format(layer_idx)] = hidden_layer
                input_dim = layer_output_dim
            # Output features.
            output_layer = {}
            for ofn in _output_feature_names:
                if ofn in _sparse_feature_names:
                    feature_dim = len(sparse_feature_vocabs[ofn])
                else:
                    feature_dim = 1
                w = tf.get_variable(
                    'output_w_{}'.format(ofn),
                    shape=[input_dim, feature_dim],
                    initializer=tf.truncated_normal_initializer(
                        stddev=1.0 / np.sqrt(feature_dim)))
                if inspect:
                    self._vars_to_inspect['output_w_{}'.format(ofn)] = w
                if use_bias:
                    bias = tf.get_variable(
                        'output_bias_{}'.format(ofn),
                        shape=[1, feature_dim],
                        initializer=tf.truncated_normal_initializer(
                            stddev=1.0 / np.sqrt(feature_dim)))
                    if inspect:
                        self._vars_to_inspect[
                            'output_bias_{}'.format(ofn)] = bias
                else:
                    bias = tf.constant(0.0, shape=[1, feature_dim])
                output_layer[ofn] = {
                    'labels': features[ofn],
                    'logits': tf.add(tf.matmul(hidden_layer, w), bias)
                    # w / 10.), bias)
                }
                if inspect:
                    self._vars_to_inspect[
                        f'output_labels_{ofn}'] = output_layer[ofn]['labels']
                    self._vars_to_inspect[
                        f'output_logits_{ofn}'] = output_layer[ofn]['logits']
            return hidden_layer, output_layer

        def similarity_loss(top_embeddings, output_layer):
            """Build the loss to be optimized.

            Args:
              top_embeddings: First element returned by create_tower.
              output_layer: Second element returned by create_tower.

            Returns:
              total_loss: A tensor of shape [1] with the total loss
                to be optimized.
              losses: A dictionary keyed by output feature names,
                of tensors of shape [1] with the contribution
                to the loss of each output feature.
            """
            losses = {}
            total_loss = tf.scalar_mul(
                _l2_regularization, tf.nn.l2_loss(top_embeddings))
            for _fn, output in output_layer.items():
                if _fn in _sparse_feature_names:
                    losses[_fn] = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=output['logits'],
                            labels=tf.feature_to_id(
                                output['labels'],
                                keys_and_values=keys_and_values[_fn])))
                else:
                    losses[_fn] = tf.sqrt(tf.reduce_mean(tf.square(
                        output['logits'] - tf.cast(
                            output['labels'], tf.float32))))
                total_loss += losses[_fn]
            return total_loss, losses

        # Body of the constructor.
        input_feature_columns = generate_feature_columns(_input_feature_names)
        # Train
        with tf.variable_scope('model', reuse=False):
            train_hidden_layer, train_output_layer = create_tower(
                make_batch(train_dataframe, batch_size), input_feature_columns)
            self._train_loss, train_losses = similarity_loss(
                train_hidden_layer, train_output_layer)
        # Test
        with tf.variable_scope('model', reuse=True):
            test_hidden_layer, test_output_layer = create_tower(
                make_batch(test_dataframe, batch_size), input_feature_columns)
            test_loss, test_losses = similarity_loss(
                test_hidden_layer, test_output_layer)
        # Whole dataframe to get final embeddings
        with tf.variable_scope('model', reuse=True):
            self._hidden_layer, _ = create_tower(
                make_batch(dataframe, dataframe.shape[0]),
                input_feature_columns)
        # Metrics is a dictionary of dictionaries of dictionaries. The 3 levels
        # are used as plots, line colors, and line styles respectively.
        self._metrics = {
            'total': {'train': {'loss': self._train_loss},
                      'test': {'loss': test_loss}},
            'feature': {
                'train': {f'{_k} loss': _v for _k, _v in train_losses.items()},
                'test': {f'{_k} loss': _v for _k, _v in test_losses.items()}}}

    def train(self, num_iterations=30, learning_rate=1.0, plot_results=True,
              optimizer=tf.train.GradientDescentOptimizer):
        """Trains the model.

        Args:
          num_iterations: int, the number of iterations to run.
          learning_rate: float, the optimizer learning rate.
          plot_results: bool, whether to plot the results at the end of training.
          optimizer: tf.train.Optimizer, the optimizer to be used for training.
        """
        with self._train_loss.graph.as_default():
            opt = optimizer(learning_rate)
            train_op = opt.minimize(self._train_loss)
            opt_init_op = tf.variables_initializer(opt.variables())
            if self._session is None:
                self._session = tf.Session()
                with self._session.as_default():
                    self._session.run(tf.global_variables_initializer())
                    self._session.run(tf.local_variables_initializer())
                    self._session.run(tf.tables_initializer())

        with self._session.as_default():
            self._session.run(opt_init_op)
            if plot_results:
                iterations = []
                metrics_vals = {k0: {k1: {k2: [] for k2 in v1}
                                     for k1, v1 in v0.items()}
                                for k0, v0 in self._metrics.items()}

            # Train and append results.
            for i in range(num_iterations + 1):
                _, results = self._session.run((train_op, self._metrics))

                # Printing the 1 liner with losses.
                if i % 10 == 0 or i == num_iterations:
                    print()
                    print(f'iteration{i:5d},  ' + ',  '.join(
                        [f'{k0} {k1} {k2}: {v2:7.3f}'
                         for k0, v0 in results.items()
                         for k1, v1 in v0.items()
                         for k2, v2 in v1.items()]), end=' ')
                    if plot_results:
                        iterations.append(i)
                        for k0, v0 in results.items():
                            for k1, v1 in v0.items():
                                for k2, v2 in v1.items():
                                    metrics_vals[k0][k1][k2].append(
                                        results[k0][k1][k2])

            # Feedforward the entire dataframe to get all the embeddings.
            self._embeddings = self._session.run(self._hidden_layer)

            # Plot the losses and embeddings.
            if plot_results:
                num_subplots = len(metrics_vals) + 1
                colors = 10 * ('red', 'blue', 'black', 'green')
                styles = 10 * ('-', '--', '-.', ':')
                # Plot the metrics.
                fig = plt.figure()
                fig.set_size_inches(num_subplots * 10, 8)
                for i0, (k0, v0) in enumerate(metrics_vals.items()):
                    ax = fig.add_subplot(1, num_subplots, i0 + 1)
                    ax.set_title(k0)
                    for i1, (k1, v1) in enumerate(v0.items()):
                        for i2, (k2, v2) in enumerate(v1.items()):
                            ax.plot(iterations, v2, label=f'{k1} {k2}',
                                    color=colors[i1], linestyle=styles[i2])
                    ax.set_xlim([1, num_iterations])
                    ax.set_yscale('log')
                    ax.legend()
                # Plot the embeddings (first 3 dimensions).
                ax.legend(loc='upper right')
                ax = fig.add_subplot(1, num_subplots, num_subplots)
                ax.scatter(self._embeddings[:, 0], self._embeddings[:, 1],
                           alpha=0.5, marker='o')
                ax.set_title('embeddings')
                plt.show()

    @property
    def embeddings(self):
        return self._embeddings


# Define some constants related to this dataset.
sparse_feature_names = ('maker', 'maker_location', 'broad_origin',
                        'specific_origin', 'bean_type')
dense_feature_names = ('reference_number', 'review_date', 'cocoa_percent',
                       'rating')
# Set of features used as input to the similarity model.
input_feature_names = ('maker', 'maker_location', 'broad_origin',
                       'cocoa_percent', 'bean_type', 'rating', )
# Set of features used as output to the similarity model.
output_feature_names = ['rating']

# As a rule of thumb, a reasonable choice for the embedding dimension of a
# sparse feature column is the log2 of the cardinality of its vocabulary.
# sparse_input_feature_embedding_dims = { 'maker': 9, 'maker_location': 6, ... }
default_embedding_dims = {
    sfn: int(round(np.log(choc_data[sfn].nunique()) / np.log(2)))
    for sfn in set(sparse_feature_names).intersection(input_feature_names)
}
# Dictionary mapping each sparse input feature to the dimension
# of its embedding space.
sparse_input_feature_embedding_dims = default_embedding_dims

# Weight of the L2 regularization applied to the top embedding layer.
l2_regularization = 1
# List of dimensions of the hidden layers of the deep neural network.
hidden_dims = [20, 10]

print('------ build model')
with tf.Graph().as_default():
    similarity_model = SimilarityModel(
        choc_data,
        _input_feature_names=input_feature_names,
        _output_feature_names=output_feature_names,
        _dense_feature_names=dense_feature_names,
        _sparse_input_feature_embedding_dims=sparse_input_feature_embedding_dims,
        _hidden_dims=hidden_dims,
        _l2_regularization=l2_regularization,
        batch_size=100,
        use_bias=True,
        inspect=True)

print('------ train model')
similarity_model.train(
    num_iterations=1000,
    learning_rate=0.1,
    optimizer=tf.train.AdagradOptimizer)
print()


# Clustering chocolate dataset

k = 160
choc_embed = similarity_model.embeddings
choc_embed = pd.DataFrame(choc_embed)
choc_feature_cols = choc_embed.columns.values  # save original columns
# initialize every point to an impossible value, the k+1 cluster
choc_embed['centroid'] = k
# init the point to centroid distance to an impossible value '2' (>1)
choc_embed['pt2centroid'] = 2
[choc_embed, centroids] = clustering_utils.kmeans(
    choc_embed, k, choc_feature_cols, 1)
print('Data for the first few chocolates, ' 
      'with "centroid" and "pt2centroid" on the extreme right:')
print(choc_embed)
print()


# Inspect clustering result

cluster_number = 20
print(choc_data.loc[choc_embed['centroid'] == cluster_number, :])
print()


# Quality metrics for clusters

clustering_utils.cluster_quality_metrics(choc_embed)

kmin = 10
kmax = 200
kstep = 1
clustering_utils.loss_vs_clusters(
    kmin, kmax, kstep, choc_embed, choc_feature_cols)
