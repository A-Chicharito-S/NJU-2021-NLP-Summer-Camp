import numpy as np
import settings
import collections


# create_vocabulary() deals data and count words as well as their frequency
# returns **a list for vocabulary**, **a list for words' frequency**,
# **a dictionary maintaining (word<-->frequency) pairs**, **sentence maxlen**
def create_vocabulary():
    word_list = []
    maxlen = 0
    with open(settings.NEG_DATA, 'r', encoding='UTF-8') as f:
        f_lines = f.readlines()  # still don't know if this encoding, decode() works
        for line in f_lines:
            words = line.strip().split()
            maxlen = max(maxlen, len(words))
            word_list.extend(words)
    with open(settings.POS_DATA, 'r', encoding='UTF-8') as f:
        f_lines = f.readlines()
        for line in f_lines:
            words = line.strip().split()
            maxlen = max(maxlen, len(
                words))
            # need to experiment if words able to be len(), and len(words) is actually the length of the sentence
            word_list.extend(words)
    counter = collections.Counter(word_list)  # count word frequency
    sorted_words = sorted(counter.items(), key=lambda x: x[1],
                          reverse=True)  # frequency high to low, returns a dictionary object
    word_list = [word[0] for word in sorted_words]
    word_freq = [word[1] for word in sorted_words]
    word_list = ['<unk>'] + word_list[:settings.VOCAB_SIZE - 1]
    word_freq = [0] + word_freq[:settings.VOCAB_SIZE - 1]
    word_dict = dict(zip(word_list, word_freq))  # word_dict stores (key: word<-->value: word_frequency) pairs
    return word_list, word_freq, word_dict, maxlen


# create_word_embedding_dict(word_dict) reads a **dictionary** with words as keys
# and returns a dictionary with the same words as keys and corresponding word embeddings as values
def create_word_embedding_dict(word_dict):
    word_embedding_dict = {}
    with open(settings.WORD_EMBEDDING, 'r', encoding='UTF-8') as f:
        f_lines = f.readlines()
        for line in f_lines:
            word = line.strip().split()[0]
            # need to be cautious here since now no knowledge about the format of the lines of the word embedding
            # strip() removes the backspace in the front and back of the string
            if word in word_dict.keys():  # can also use haskey() method
                embedding = list(map(float, list([float(num) for num in line.strip(word + ' ').split()])))

                # still note that whether GloVe has embedding for <unk> token!!!,if so, codes need to change
                # here remember that if the str is like: '2,3,4', list will be ['2', ',', '3', ',', '4'],
                # thus , should be removed
                # list(map(type, list)) converts numeric string into list with numbers of type 'type' as members
                word_embedding_dict[word] = embedding

    return word_embedding_dict


# make_dataset(word_embedding_dict) makes use of the word_embedding_dict and then returns **data**
# and corresponding **label** (in the form of NumPy array). non-shuffled, non-padded
def make_dataset(word_embedding_dict, maxlen):
    data = []
    label = []
    unk = [0.0] * settings.WORD_EMBEDDING_SIZE  # Is it all zeros??since padding is zero
    padding = [0.0] * settings.WORD_EMBEDDING_SIZE  # used for padding

    with open(settings.NEG_DATA, 'r', encoding='UTF-8') as f:
        f_lines = f.readlines()
        for line in f_lines:
            words = line.strip().split()
            sentence = []
            for word in words:
                if word in word_embedding_dict.keys():
                    sentence.append(word_embedding_dict[word])
                else:
                    sentence.append(unk)
            for _ in range(maxlen - len(words)):
                sentence.append(padding)
            label.append(0)
            data.append(sentence)

    with open(settings.POS_DATA, 'r', encoding='UTF-8') as f:
        f_lines = f.readlines()
        for line in f_lines:
            words = line.strip().split()
            sentence = []
            for word in words:
                if word in word_embedding_dict.keys():
                    sentence.append(word_embedding_dict[word])
                else:
                    sentence.append(unk)
            for _ in range(maxlen - len(words)):
                sentence.append(padding)
            label.append(1)
            data.append(sentence)
    data = np.array(data, dtype=float)
    label = np.array(label, dtype=float)
    return data, label


if __name__ == "__main__":
    word_list, word_frequency, word_dict, maxlen = create_vocabulary()
    word_embedding_dict = create_word_embedding_dict(word_dict)
    data, label = make_dataset(word_embedding_dict, maxlen)
    print('data shape {}'.format(data.shape))
    print('data {}'.format(data[1, 0:10, :]))
    print('label shape {}'.format(label.shape))
    print('label {}'.format(label))
