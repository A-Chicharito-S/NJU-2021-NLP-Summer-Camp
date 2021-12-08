import keras
NEG_DATA = 'rt-polarity.neg'
POS_DATA = 'rt-polarity.pos'
VOCAB_SIZE = 10000  # vocabulary size
WORD_EMBEDDING_SIZE = 200  # this variable may be used in WORD_EMBEDDING to specify the word embedding size
WORD_EMBEDDING = 'glove.6B.' + str(WORD_EMBEDDING_SIZE) + 'd.txt'
LR = 0.01
LR_DECAY = 0.9
DECAY_STEP = 10000
MOMENTUM = 0.7
BATCH_SIZE = 64
EPOCH = 2000
CHECKPOINT_PATH = 'check.hdf5'
MIN_DELTA = 0.001
PATIENCE = 50
HISTORY = keras.callbacks.History()
