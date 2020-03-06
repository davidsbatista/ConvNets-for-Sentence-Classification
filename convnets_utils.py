import os
import numpy as np

from sklearn.metrics import classification_report

from keras.models import Model
from keras.activations import relu
from keras.layers import Input, Dense, Embedding, Flatten, Conv1D, MaxPooling1D
from keras.layers import Dropout, concatenate


def load_fasttext_embeddings():
    glove_dir = '/Users/dsbatista/resources/glove.6B'
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index


def create_embeddings_matrix(embeddings_index, vocabulary, embedding_dim=100):
    embeddings_matrix = np.random.rand(len(vocabulary) + 1, embedding_dim)
    for i, word in enumerate(vocabulary):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
    print('Matrix shape: {}'.format(embeddings_matrix.shape))
    return embeddings_matrix


def get_embeddings_layer(embeddings_matrix, name, max_len, trainable=False):
    embedding_layer = Embedding(
        input_dim=embeddings_matrix.shape[0],
        output_dim=embeddings_matrix.shape[1],
        input_length=max_len,
        weights=[embeddings_matrix],
        trainable=trainable,
        name=name)
    return embedding_layer


def get_conv_pool(x_input, max_len, suffix, n_grams=[3, 4, 5], feature_maps=100):
    branches = []
    for n in n_grams:
        branch = Conv1D(filters=feature_maps, kernel_size=n, activation=relu,
                        name='Conv_' + suffix + '_' + str(n))(x_input)
        branch = MaxPooling1D(pool_size=max_len - n + 1, strides=None, padding='valid',
                              name='MaxPooling_' + suffix + '_' + str(n))(branch)
        branch = Flatten(name='Flatten_' + suffix + '_' + str(n))(branch)
        branches.append(branch)
    return branches


def get_cnn_rand(embedding_dim, vocab_size, max_len, num_classes, loss='categorical_crossentropy'):
    # create the embedding layer
    embedding_matrix = np.random.rand(vocab_size, embedding_dim)
    embedding_layer = get_embeddings_layer(embedding_matrix, 'embedding_layer_dynamic',
                                           max_len, trainable=True)

    # connect the input with the embedding layer
    i = Input(shape=(max_len,), dtype='int32', name='main_input')
    x = embedding_layer(i)

    # generate several branches in the network, each for a different convolution+pooling operation,
    # and concatenate the result of each branch into a single vector
    branches = get_conv_pool(x, max_len, 'dynamic')
    z = concatenate(branches, axis=-1)
    z = Dropout(0.5)(z)

    # pass the concatenated vector to the prediction layer
    o = Dense(num_classes, activation='softmax', name='output')(z)

    model = Model(inputs=i, outputs=o)
    model.compile(loss={'output': loss}, optimizer='adam', metrics=['accuracy'])

    return model


def get_cnn_pre_trained_embeddings(embedding_layer, max_len, num_classes,
                                   loss='categorical_crossentropy'):
    # connect the input with the embedding layer
    i = Input(shape=(max_len,), dtype='int32', name='main_input')
    x = embedding_layer(i)

    # generate several branches in the network, each for a different convolution+pooling operation,
    # and concatenate the result of each branch into a single vector
    branches = get_conv_pool(x, max_len, 'static')
    z = concatenate(branches, axis=-1)
    z = Dropout(0.5)(z)

    # pass the concatenated vector to the prediction layer
    o = Dense(num_classes, activation='softmax', name='output')(z)

    model = Model(inputs=i, outputs=o)
    model.compile(loss={'output': loss}, optimizer='adam', metrics=['accuracy'])

    return model


def get_cnn_multichannel(embedding_layer_channel_1, embedding_layer_channel_2, max_len,
                         num_classes, loss='categorical_crossentropy'):
    # dynamic channel
    input_dynamic = Input(shape=(max_len,), dtype='int32', name='input_dynamic')
    x = embedding_layer_channel_1(input_dynamic)
    branches_dynamic = get_conv_pool(x, max_len, 'static')
    z_dynamic = concatenate(branches_dynamic, axis=-1)

    # static channel
    input_static = Input(shape=(max_len,), dtype='int32', name='input_static')
    x = embedding_layer_channel_2(input_static)
    branches_static = get_conv_pool(x, max_len, 'dynamic')
    z_static = concatenate(branches_static, axis=-1)

    # concatenate both models and pass to classification layer
    z = concatenate([z_static, z_dynamic], axis=-1)
    z = Dropout(0.5)(z)

    # pass the concatenated vector to the prediction layer
    o = Dense(num_classes, activation='softmax', name='output')(z)

    model = Model(inputs=[input_dynamic, input_static], outputs=o)
    model.compile(loss={'output': loss}, optimizer='adam', metrics=['accuracy'])

    return model
