import keras.backend
from keras.models import Model
from keras.layers import Input, Dense, Bidirectional, LSTM, Concatenate, Flatten, Softmax, Dropout
import settings
from keras.engine.topology import Layer
import tensorflow as tf

"""
# or maybe just return two??---currently don't know how to do it
# this should be defined as a layer otherwise it's not callable
def make_att(arr_1, arr_2):  # takes arr_1 to calculate the average and attentions arr_2
    avr = keras.backend.mean(arr_1, axis=0, keepdims=True)  # now avr is a row (1 x settings.LSTM_HIDDEN_SIZE)
    avr_TransP = keras.backend.transpose(avr)
    # is it ok to use np.mean()? can keepdims set to be False?
    W_a = tf.Variable(tf.random.normal(shape=[settings.LSTM_HIDDEN_STATES, settings.LSTM_HIDDEN_STATES],
                                       mean=0, stddev=1), trainable=True, name='W_a')
    b_a = tf.Variable(tf.random.normal(shape=[arr_2.shape[0], 1],
                                       mean=0, stddev=1), trainable=True, name='b_a')
    temp_1 = keras.backend.dot(arr_2, W_a)
    temp_2 = keras.backend.dot(temp_1, avr_TransP)
    att_raw = keras.backend.tanh(temp_2+b_a)
    att = keras.backend.softmax(keras.backend.transpose(att_raw))  # now att is (arr_2.shape[0] x 1)
    out = keras.backend.dot(att, arr_2)  # now out is (1 x arr_2.shape[0])
    return out
    # keras.backend.softmax(x) to calculate the softmax of input x (only use raw as computing unit if a 2-d matrix)
"""
"""
class MakeAtt(Layer):
    def __init__(self, **kwargs):
        super(MakeAtt, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        _, arr_2_shape = input_shape
        # 为该层创建一个可训练的权重
        self.W_a = self.add_weight(name='W_a',
                                   shape=(settings.LSTM_HIDDEN_STATES, settings.LSTM_HIDDEN_STATES),
                                   initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                                   trainable=True)
        self.b_a = self.add_weight(name='b_a', shape=(arr_2_shape[0], 1),
                                   initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                                   trainable=True)

        super(MakeAtt, self).build(input_shape)
        # 一定要在最后调用它, here 'input_shape' may just be the parameters to specify which build() method
        # may equal to: self.built = True

    def call(self, x):  # x can be considered a list here!
        assert isinstance(x, list)
        arr_1, arr_2 = x
        avr = keras.backend.mean(arr_1, axis=0, keepdims=True)  # now avr is a row (1 x settings.LSTM_HIDDEN_SIZE)
        avr_TransP = keras.backend.transpose(avr)
        # is it ok to use np.mean()? can keepdims set to be False?
        temp_1 = keras.backend.dot(arr_2, self.W_a)
        temp_2 = keras.backend.dot(temp_1, avr_TransP)
        att_raw = keras.backend.tanh(temp_2 + self.b_a)
        att = keras.backend.softmax(keras.backend.transpose(att_raw))  # now att is (arr_2.shape[0] x 1)
        out = keras.backend.dot(att, arr_2)  # now out is (1 x arr_2.shape[0])
        return out

    def compute_output_shape(self, input_shape):
        return (1 , input_shape[1][0])
"""


class MakeAtt(Layer):
    def __init__(self, max_len_of_second, **kwargs):
        super(MakeAtt, self).__init__(**kwargs)
        self.len = max_len_of_second
        self.b_a = self.add_weight(name='b_a', shape=(self.len, 1),
                                   initializer=keras.initializers.Zeros(),
                                   trainable=True)
        self.W_a = self.add_weight(name='W_a',
                                   shape=(settings.LSTM_HIDDEN_STATES*2, settings.LSTM_HIDDEN_STATES*2),
                                   initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                   trainable=True)

    def call(self, inputs):  # x can be considered a list here!
        assert isinstance(inputs, list)
        arr_1, arr_2 = inputs
        avr = tf.reduce_mean(arr_1, axis=1)  # now avr is a row (1 x settings.LSTM_HIDDEN_SIZE)
        avr_TransP = tf.expand_dims(avr, -1)
        # print(avr.shape)
        # print(avr_TransP.shape)
        # print(arr_2.shape)
        # is it ok to use np.mean()? can keepdims set to be False?
        temp_1 = tf.matmul(arr_2, self.W_a)
        # print('temp1:{}'.format(temp_1.shape))
        temp_2 = tf.matmul(temp_1, avr_TransP)
        # print('temp2:{}'.format(temp_2.shape))
        att_raw = tf.tanh(temp_2 + self.b_a)
        att_squeeze = tf.squeeze(att_raw, [2])
        # print('att_raw:{}'.format(att_raw.shape))
        # print('after squeeze{}'.format(att_squeeze))
        att = tf.nn.softmax(tf.expand_dims(att_squeeze, axis=1), axis=2)  # now att is (1 x arr_2.shape[0])
        # print('att:{}'.format(att.shape))
        out = tf.matmul(att, arr_2)  # now out is (1 x arr_2.shape[0])
        # print('shape of out:{}'.format(out.shape))
        return out, att


def make_model(max_S_len, max_T_len):
    input_s = Input(shape=(max_S_len, settings.WORD_EMBEDDING_SIZE))
    input_t = Input(shape=(max_T_len, settings.WORD_EMBEDDING_SIZE))
    lstm_s = Bidirectional(LSTM(units=settings.LSTM_HIDDEN_STATES, return_sequences=True,
                                kernel_regularizer=keras.regularizers.l2(settings.LAMBDA_r),
                                bias_regularizer=keras.regularizers.l2(settings.LAMBDA_r)))(input_s)
    # print('lstm_s shape{}'.format(lstm_s.shape))
    # lstm_s's shape: (max_S_len x settings.LSTM_HIDDEN_STATES)
    lstm_t = Bidirectional(LSTM(units=settings.LSTM_HIDDEN_STATES, return_sequences=True,
                                kernel_regularizer=keras.regularizers.l2(settings.LAMBDA_r),
                                bias_regularizer=keras.regularizers.l2(settings.LAMBDA_r)))(input_t)
    # consider adding recurrent_dropout=x for both LSTMs
    # print('lstm_t shape{}'.format(lstm_t.shape))
    # lstm_t's shape: (max_T_len x settings.LSTM_HIDDEN_STATES)
    T_r, T_att = MakeAtt(max_len_of_second=max_T_len)([lstm_s, lstm_t])
    # attentions lstm_t
    C_r, C_att = MakeAtt(max_len_of_second=max_S_len)([lstm_t, lstm_s])
    # attentions lstm_s (context)
#    T_r = Lambda(make_att, arguments={'arr_2': lstm_t})(lstm_s)  # attentions lstm_t
#    C_r = Lambda(make_att, arguments={'arr_2': lstm_s})(lstm_t)  # attentions lstm_s
    concat = Concatenate(axis=2)([T_r, C_r])  # now concat is a (1 x n) tensor
    input_dense = Flatten()(concat)
    output_dense = Dense(settings.CATEGORY, activation='tanh',
                         kernel_regularizer=keras.regularizers.l2(settings.LAMBDA_r),
                         bias_regularizer=keras.regularizers.l2(settings.LAMBDA_r))(input_dense)
    # a 3-classification problem
    output_dropout = Dropout(0.5)(output_dense)
    output_final = Softmax()(output_dropout)
    model = Model(inputs=[input_s, input_t], outputs=output_final)
    return model




