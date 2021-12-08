from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Reshape
import settings
import keras


# should know about the datasize: (number of sentences x max_sentence_len x word_embedding_size)
# also think about whether we need to divide data by (max_sentence_len x word_embedding_size -1)
# just like '/255' operation

# construct_model(maxlen) takes maxlen of the sentences in the dataset as input
# and returns the model for training, testing
def construct_model(maxlen):
    input = Input(
        shape=(maxlen, settings.WORD_EMBEDDING_SIZE, 1))  # remember to reshape the input data in train_eval.py!!!

    output_height3 = Conv2D(100, [3, settings.WORD_EMBEDDING_SIZE],
                            input_shape=(maxlen, settings.WORD_EMBEDDING_SIZE, 1))(input)
    # need to know if any other parametres can be initialized to achieve better performance

    o_h3_maxP = MaxPooling2D(pool_size=(maxlen - 2, 1))(output_height3)
    o_h3_maxP = Reshape((100, 1))(o_h3_maxP)

    output_height4 = Conv2D(100, [4, settings.WORD_EMBEDDING_SIZE],
                            input_shape=(maxlen, settings.WORD_EMBEDDING_SIZE, 1))(input)
    o_h4_maxP = MaxPooling2D(pool_size=(maxlen - 3, 1))(output_height4)
    o_h4_maxP = Reshape((100, 1))(o_h4_maxP)

    output_height5 = Conv2D(100, [5, settings.WORD_EMBEDDING_SIZE],
                            input_shape=(maxlen, settings.WORD_EMBEDDING_SIZE, 1))(input)
    o_h5_maxP = MaxPooling2D(pool_size=(maxlen - 4, 1))(output_height5)
    o_h5_maxP = Reshape((100, 1))(o_h5_maxP)

    concat = keras.layers.merge.Concatenate(axis=1)([o_h3_maxP, o_h4_maxP,
                                                     o_h5_maxP])
    # now the size of concat should be batch_size x 300 (3 different size of kernels with 100 each)

    concat_flat = keras.layers.Flatten()(concat)
    output_penultimate = Dense(1024, activation='relu')(concat_flat)
    output_final = Dense(2, activation='sigmoid')(output_penultimate)

    model = Model(inputs=input, outputs=output_final)

    return model
