LAPTOPS_TRAIN = 'data/laptop/train.txt'
LAPTOPS_TEST = 'data/laptop/test.txt'
RESTAURANTS_TRAIN = 'data/restaurant/train.txt'
RESTAURANTS_TEST = 'data/restaurant/test.txt'
LAPTOPS_TEST_WORDS = 'data_processed/laptops_text_words.txt'
RESTAURANTS_TEST_WORDS = 'data_processed/restaurants_text_words.txt'
LAPTOPS_TEST_PROCESSED = 'data_processed/laptops_text_processed.txt'
RESTAURANTS_TEST_PROCESSED = 'data_processed/restaurants_text_processed.txt'
VOCAB_SIZE = 4000
# lap: total number of words: 4203, vocab is set to 4000, word_embedding_size=300
# training sentence: 2328
# testing sentence: 642
# max context length: 83, max target length: 8

# rest: total number of words: 5276, vocab is set to 5000, word_embedding_size=300
# training sentence: 3699
# testing sentence: 1134
# max context length: 78, max target length: 21

WORD_EMBEDDING_SIZE = 300
WORD_EMBEDDING = 'word_embeddings/glove.6B.' + str(WORD_EMBEDDING_SIZE) + 'd.txt'  # where the Glove vector stores
LSTM_HIDDEN_STATES = 300
CATEGORY = 3
TEST_SPLIT = 0.2
LR = 0.05
MOMENTUM = 0.9
MIN_DELTA = 0.001
PATIENCE = 10
EPOCH = 50
BATCH_SIZE = 64
LAMBDA_r = 0.00001
