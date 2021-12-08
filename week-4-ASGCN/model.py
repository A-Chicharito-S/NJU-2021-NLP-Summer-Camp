import keras.backend
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Bidirectional, LSTM, Concatenate, Flatten, Softmax, Dropout, Masking
import settings
from keras.engine.topology import Layer
import tensorflow as tf


class GCN(Layer):
    def __init__(self, layer_num, sentence_len, hidden_size, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.len = sentence_len
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.Ws, self.bs = self.make_weights()  # W: sentence_len x sentence_len, b: sentence_len x 1

    def call(self, inputs):  # x can be considered a list here!
        assert isinstance(inputs, list)
        self.h, self.mask, self.a, self.weight_q = inputs
        # print(self.h)
        # print(self.mask)
        # print(self.a)
        # self.weight_q = self.compute_q(input_mask=self.mask)
        # lstm_seq: batch size x sentence_len x self.hidden_size (= 2 * LSTM hidden size)
        # mask: batch size x sentence_len x LSTM hidden size
        # a: batch size x sentence_len x sentence_len
        for i in range(self.layer_num):
            if i == 0:
                self.h = self.h
            else:
                self.h = self.h * self.weight_q
            # shape: batch size x sentence_len x self.hidden_size
            self.h = tf.matmul(tf.matmul(self.a, self.Ws[i]), self.h)  ##may have errors since Ws have no batch size!!!
            # tf.matmul(self.a, self.Ws[0]) shape: batch size x sentence_len x sentence_len
            # self.h shape: batch size x sentence_len x self.hidden_size
            d = tf.reduce_sum(self.a, axis=2) + 1  # shape: batch size x sentence_len
            d = tf.tile(tf.expand_dims(d, axis=-1), [1, 1, self.hidden_size])

            self.h = tf.nn.relu(self.h / d + self.bs[i])

        return self.h
        # shape: batch size x sentence_len x self.hidden_size

    def make_weights(self):
        W = []
        b = []
        for i in range(self.layer_num):
            W.append(self.add_weight(name='W' + str(i + 1),
                                     shape=(self.len, self.len),
                                     initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                     regularizer=keras.regularizers.l2(settings.LAMBDA_r),
                                     trainable=True))
            b.append(self.add_weight(name='b' + str(i + 1), shape=(self.len, 1),
                                     initializer=keras.initializers.Zeros(),
                                     regularizer=keras.regularizers.l2(settings.LAMBDA_r),
                                     trainable=True))
        return W, b


class MaskAndAtt(Layer):
    def __init__(self, **kwargs):
        super(MaskAndAtt, self).__init__(**kwargs)

    def call(self, inputs):  # x can be considered a list here!
        assert isinstance(inputs, list)
        self.h, self.lstm, self.mask = inputs
        # h: batch size x sentence_len x self.hidden_size (= 2 * LSTM hidden size)
        # lstm: batch size x sentence_len x (2 * LSTM hidden size)
        # mask: batch size x sentence_len x LSTM hidden size
        self.mask = tf.tile(self.mask, [1, 1, 2])
        self.h = self.h * self.mask  # shape: batch size x sentence_len x self.hidden_size (= 2 * LSTM hidden size)
        beta = tf.matmul(self.lstm, tf.transpose(self.h, perm=[0, 2, 1]))
        beta = tf.reduce_sum(beta, axis=-1)  # shape: batch size x sentence_len
        alpha = tf.nn.softmax(beta, axis=-1)
        alpha = tf.expand_dims(alpha, axis=1)  # shape: batch size x 1 x sentence_len
        r = tf.matmul(alpha, self.lstm)  # shape: batch size x 1 x (2 * LSTM hidden size)
        return r, alpha


def make_model(sentence_len):
    input_s = Input(shape=(sentence_len, settings.WORD_EMBEDDING_SIZE))
    input_m = Input(shape=(sentence_len, settings.WORD_EMBEDDING_SIZE))
    input_a = Input(shape=(sentence_len, sentence_len))
    input_q = Input(shape=(sentence_len, 2 * settings.LSTM_HIDDEN_STATES))
    # print('type for input_s: {}'.format(type(input_s)))

    output_s_from_mask = Masking(mask_value=0.0)(input_s)
    # print(output_s_from_mask)
    output_s_from_lstm = Bidirectional(
        LSTM(units=settings.LSTM_HIDDEN_STATES, return_sequences=True,
             kernel_regularizer=keras.regularizers.l2(settings.LAMBDA_r),
             bias_regularizer=keras.regularizers.l2(settings.LAMBDA_r)))(output_s_from_mask)

    output_s_from_GCN = GCN(layer_num=2, sentence_len=sentence_len, hidden_size=2 * settings.LSTM_HIDDEN_STATES)(
        [output_s_from_lstm, input_m, input_a, input_q])
    # shape: batch size x sentence_len x 2 * settings.LSTM_HIDDEN_STATES
    output_s_from_Att, _ = MaskAndAtt()([output_s_from_GCN, output_s_from_lstm, input_m])
    output_s_from_Flatten = Flatten()(output_s_from_Att)
    output_dense = Dense(settings.CATEGORY, activation='tanh',
                         kernel_regularizer=keras.regularizers.l2(settings.LAMBDA_r),
                         bias_regularizer=keras.regularizers.l2(settings.LAMBDA_r))(output_s_from_Flatten)
    # a 3-classification problem
    output_dropout = Dropout(0.5)(output_dense)
    output_final = Softmax()(output_dropout)
    model = Model(inputs=[input_s, input_m, input_a, input_q], outputs=output_final)
    return model
