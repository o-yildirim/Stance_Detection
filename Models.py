from keras.models import  Model
from keras.layers import *
from keras import regularizers
from keras.initializers import Constant
import tensorflow_addons as tfa
import tensorflow as tf

def get_Kims_model_updated(vocab_size, embedding_dim, embedding_matrix, input_max_length):
    num_filters = 256

    inputs = Input(shape=(input_max_length,), dtype='int32')

    embedding_layer = Embedding(vocab_size,
                                embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=input_max_length,
                                trainable=False)(inputs)
    # trainable=True)(inputs)

    reshape = Reshape((input_max_length, embedding_dim, 1))(embedding_layer)

    conv_0 = Conv2D(num_filters, kernel_size=(2, embedding_dim), activation='relu',
                    kernel_regularizer=regularizers.l2(2))(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(3, embedding_dim), activation='relu',
                    kernel_regularizer=regularizers.l2(2))(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(5, embedding_dim), activation='relu',
                    kernel_regularizer=regularizers.l2(2))(reshape)

    bn_0 = BatchNormalization()(conv_0)
    bn_1 = BatchNormalization()(conv_1)
    bn_2 = BatchNormalization()(conv_2)

    maxpool_0 = MaxPool2D(pool_size=(input_max_length - 2 + 1, 1), strides=(1, 1), padding='valid')(bn_0)
    maxpool_1 = MaxPool2D(pool_size=(input_max_length - 3 + 1, 1), strides=(1, 1), padding='valid')(bn_1)
    maxpool_2 = MaxPool2D(pool_size=(input_max_length - 5 + 1, 1), strides=(1, 1), padding='valid')(bn_2)

    concatenated_tensor_2 = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten_2 = Flatten()(concatenated_tensor_2)
    dropout = Dropout(0.6)(flatten_2)
    output = Dense(units=2, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  metrics=[tfa.metrics.F1Score(num_classes=2, average='macro')])

    model.summary()
    return model


def get_Kims_model_original(vocab_size, embedding_dim, embedding_matrix,
                            input_max_length):  # This code module is obtained from https://www.kaggle.com/code/hamishdickson/cnn-for-sentence-classification-by-yoon-kim => Static version
    num_filters = 100

    inputs = Input(shape=(input_max_length,), dtype='int32')

    embedding_layer = Embedding(vocab_size,
                                embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=input_max_length,
                                trainable=False)(inputs)

    reshape = Reshape((input_max_length, embedding_dim, 1))(embedding_layer)

    conv_0 = Conv2D(num_filters, kernel_size=(3, embedding_dim), activation='relu',
                    kernel_regularizer=regularizers.l2(3))(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(4, embedding_dim), activation='relu',
                    kernel_regularizer=regularizers.l2(3))(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(5, embedding_dim), activation='relu',
                    kernel_regularizer=regularizers.l2(3))(reshape)

    maxpool_0 = MaxPool2D(pool_size=(input_max_length - 3 + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(input_max_length - 4 + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(input_max_length - 5 + 1, 1), strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor_2 = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten_2 = Flatten()(concatenated_tensor_2)

    dropout = Dropout(0.5)(flatten_2)
    output = Dense(units=2, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=[tfa.metrics.F1Score(num_classes=2, average='macro')])
    model.summary()
    return model
