import numpy as np
import settings
import collections
import re


# create_vocabulary() deals data and count words as well as their frequency
# returns **a list for vocabulary**, **a list for words' frequency**,
# **a dictionary maintaining (word<-->frequency) pairs**, **sentence max-len(no target words)**, **target max-len**
def create_vocabulary(laptop=1, restaurant=1):
    word_list = []
    max_S_len = 0
    max_T_len = 0

    if laptop == 1:
        with open(settings.LAPTOPS_TRAIN, 'r', encoding='UTF-8') as f:
            f_lines = f.readlines()
            i = 0
            for line in f_lines:
                i = i + 1
                index = i % 4
                if index == 0:  # the 4-th row of every four row, a number sequence
                    continue
                elif index == 1:  # the first row of every four row, the target
                    words = line.strip().split()
                    max_T_len = max(max_T_len, len(words))
                    word_list.extend(words)
                elif index == 2:  # the second row of every four row, the label
                    continue
                else:  # the third row of every four row, the context
                    line = re.sub('/[0-9]*', '', line)
                    words = line.strip().split()
                    max_S_len = max(max_S_len, len(words))
                    word_list.extend(words)

        with open(settings.LAPTOPS_TEST, 'r', encoding='UTF-8') as f:
            f_lines = f.readlines()
            i = 0
            for line in f_lines:
                i = i + 1
                index = i % 4
                if index == 0:  # the 4-th row of every four row, a number sequence
                    continue
                elif index == 1:  # the first row of every four row, the target
                    words = line.strip().split()
                    max_T_len = max(max_T_len, len(words))
                    word_list.extend(words)
                elif index == 2:  # the second row of every four row, the label
                    continue
                else:  # the third row of every four row, the context
                    line = re.sub('/[0-9]*', '', line)
                    words = line.strip().split()
                    max_S_len = max(max_S_len, len(words))
                    word_list.extend(words)

    if restaurant == 1:
        with open(settings.RESTAURANTS_TRAIN, 'r', encoding='UTF-8') as f:
            f_lines = f.readlines()
            i = 0
            for line in f_lines:
                i = i + 1
                index = i % 4
                if index == 0:  # the 4-th row of every four row, a number sequence
                    continue
                elif index == 1:  # the first row of every four row, the target
                    words = line.strip().split()
                    max_T_len = max(max_T_len, len(words))
                    word_list.extend(words)
                elif index == 2:  # the second row of every four row, the label
                    continue
                else:  # the third row of every four row, the context
                    line = re.sub('/[0-9]*', '', line)
                    words = line.strip().split()
                    max_S_len = max(max_S_len, len(words))
                    word_list.extend(words)

        with open(settings.RESTAURANTS_TEST, 'r', encoding='UTF-8') as f:
            f_lines = f.readlines()
            i = 0
            for line in f_lines:
                i = i + 1
                index = i % 4
                if index == 0:  # the 4-th row of every four row, a number sequence
                    continue
                elif index == 1:  # the first row of every four row, the target
                    words = line.strip().split()
                    max_T_len = max(max_T_len, len(words))
                    word_list.extend(words)
                elif index == 2:  # the second row of every four row, the label
                    continue
                else:  # the third row of every four row, the context
                    line = re.sub('/[0-9]*', '', line)
                    words = line.strip().split()
                    max_S_len = max(max_S_len, len(words))
                    word_list.extend(words)
    counter = collections.Counter(word_list)  # count word frequency, try counter.most_common() next time
    temp = dict(counter)
    print('total number of words: {}'.format(len(temp)))
    sorted_words = sorted(temp.items(), key=lambda x: x[1], reverse=True)
    # frequency high to low, returns a dictionary object
    word_list = [word[0] for word in sorted_words]
    word_freq = [word[1] for word in sorted_words]
    word_list = ['<unk>'] + word_list[:settings.VOCAB_SIZE - 1]
    word_freq = [0] + word_freq[:settings.VOCAB_SIZE - 1]
    word_dict = dict(zip(word_list, word_freq))  # word_dict stores (key: word<-->value: word_frequency) pairs
    return word_list, word_freq, word_dict, max_S_len, max_T_len
    # previously it is max_S_len, and actually should subtract 1 since $T$ is counted as well


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
                # here remember that if the str is like: '2,3,4', list will be ['2', ',', '3', ',', '4'],
                # thus , should be removed
                # list(map(type, list)) converts numeric string into list with numbers of type 'type' as members
                word_embedding_dict[word] = embedding

    return word_embedding_dict


# make_dataset(word_embedding_dict) makes use of the word_embedding_dict and then returns **data**
# and corresponding **label** (in the form of NumPy array). non-shuffled, non-padded
def make_dataset(word_embedding_dict, max_S_len, max_T_len, laptop=1, restaurant=1):
    context_train = []
    target_train = []
    label_train = []
    context_test = []
    target_test = []
    label_test = []
    unk = np.random.uniform(-0.1, 0.1, settings.WORD_EMBEDDING_SIZE).tolist()
    padding = [0.0] * settings.WORD_EMBEDDING_SIZE

    if laptop == 1:
        with open(settings.LAPTOPS_TRAIN, 'r', encoding='UTF-8') as f:
            f_lines = f.readlines()
            i = 0
            for line in f_lines:
                i = i + 1
                index = i % 4
                if index == 0:  # the 4-th row of every four row, a number sequence
                    continue
                elif index == 1:  # the first row of every four row, the target
                    sentence_t = []
                    words = line.strip().split()
                    for word in words:
                        if word in word_embedding_dict.keys():
                            sentence_t.append(word_embedding_dict[word])
                        else:
                        #    unk = np.random.uniform(-0.1, 0.1, settings.WORD_EMBEDDING_SIZE).tolist()    #####
                            sentence_t.append(unk)
                    for _ in range(max_T_len - len(words)):
                        sentence_t.append(padding)
                    target_train.append(sentence_t)
                elif index == 2:  # the second row of every four raw, the label
                    words = line.strip().split()
                    label_train.append(int(words[0]) + 1)
                else:  # the third row of every four row, the context
                    sentence_c = []
                    line = re.sub('/[0-9]*', '', line)
                    words = line.strip().split()
                    for word in words:
                        if word in word_embedding_dict.keys():
                            sentence_c.append(word_embedding_dict[word])
                        else:
                        #    unk = np.random.uniform(-0.1, 0.1, settings.WORD_EMBEDDING_SIZE).tolist()  #####
                            sentence_c.append(unk)
                    for _ in range(max_S_len - len(words)):
                        sentence_c.append(padding)
                    context_train.append(sentence_c)

        with open(settings.LAPTOPS_TEST, 'r', encoding='UTF-8') as f:
            with open(settings.LAPTOPS_TEST_WORDS, 'w', encoding='UTF-8') as w:
                f_lines = f.readlines()
                i = 0
                for line in f_lines:
                    i = i + 1
                    index = i % 4
                    if index == 0:  # the 4-th row of every four row, a number sequence
                        continue
                    elif index == 1:  # the first row of every four row, the target
                        sentence_t = []
                        w.write(line)
                        words = line.strip().split()
                        for word in words:
                            if word in word_embedding_dict.keys():
                                sentence_t.append(word_embedding_dict[word])
                            else:
                            #    unk = np.random.uniform(-0.1, 0.1, settings.WORD_EMBEDDING_SIZE).tolist()  #####
                                sentence_t.append(unk)
                        for _ in range(max_T_len - len(words)):
                            sentence_t.append(padding)
                        target_test.append(sentence_t)
                    elif index == 2:  # the second row of every four raw, the label
                        words = line.strip().split()
                        label_test.append(int(words[0]) + 1)
                    else:  # the third row of every four row, the context
                        sentence_c = []
                        w.write(line)
                        line = re.sub('/[0-9]*', '', line)
                        words = line.strip().split()
                        for word in words:
                            if word in word_embedding_dict.keys():
                                sentence_c.append(word_embedding_dict[word])
                            else:
                            #    unk = np.random.uniform(-0.1, 0.1, settings.WORD_EMBEDDING_SIZE).tolist()  #####
                                sentence_c.append(unk)
                        for _ in range(max_S_len - len(words)):
                            sentence_c.append(padding)
                        context_test.append(sentence_c)
            w.close()
    if restaurant == 1:
        with open(settings.RESTAURANTS_TRAIN, 'r', encoding='UTF-8') as f:
            f_lines = f.readlines()
            i = 0
            for line in f_lines:
                i = i + 1
                index = i % 4
                if index == 0:  # the 4-th row of every four row, a number sequence
                    continue
                elif index == 1:  # the first row of every four row, the target
                    sentence_t = []
                    words = line.strip().split()
                    for word in words:
                        if word in word_embedding_dict.keys():
                            sentence_t.append(word_embedding_dict[word])
                        else:
                         #   unk = np.random.uniform(-0.1, 0.1, settings.WORD_EMBEDDING_SIZE).tolist()  #####
                            sentence_t.append(unk)
                    for _ in range(max_T_len - len(words)):
                        sentence_t.append(padding)
                    target_train.append(sentence_t)
                elif index == 2:  # the second row of every four raw, the label
                    words = line.strip().split()
                    label_train.append(int(words[0]) + 1)
                else:  # the third row of every four row, the context
                    sentence_c = []
                    line = re.sub('/[0-9]*', '', line)
                    words = line.strip().split()
                    for word in words:
                        if word in word_embedding_dict.keys():
                            sentence_c.append(word_embedding_dict[word])
                        else:
                         #   unk = np.random.uniform(-0.1, 0.1, settings.WORD_EMBEDDING_SIZE).tolist()  #####
                            sentence_c.append(unk)
                    for _ in range(max_S_len - len(words)):
                        sentence_c.append(padding)
                    context_train.append(sentence_c)

        with open(settings.RESTAURANTS_TEST, 'r', encoding='UTF-8') as f:
            with open(settings.RESTAURANTS_TEST_WORDS, 'w', encoding='UTF-8') as w:
                f_lines = f.readlines()
                i = 0
                for line in f_lines:
                    i = i + 1
                    index = i % 4
                    if index == 0:  # the 4-th row of every four row, a number sequence
                        continue
                    elif index == 1:  # the first row of every four row, the target
                        w.write(line)
                        sentence_t = []
                        words = line.strip().split()
                        for word in words:
                            if word in word_embedding_dict.keys():
                                sentence_t.append(word_embedding_dict[word])
                            else:
                            #    unk = np.random.uniform(-0.1, 0.1, settings.WORD_EMBEDDING_SIZE).tolist()  #####
                                sentence_t.append(unk)
                        for _ in range(max_T_len - len(words)):
                            sentence_t.append(padding)
                        target_test.append(sentence_t)
                    elif index == 2:  # the second row of every four raw, the label
                        words = line.strip().split()
                        label_test.append(int(words[0]) + 1)
                    else:  # the third row of every four row, the context
                        sentence_c = []
                        w.write(line)
                        line = re.sub('/[0-9]*', '', line)
                        words = line.strip().split()
                        for word in words:
                            if word in word_embedding_dict.keys():
                                sentence_c.append(word_embedding_dict[word])
                            else:
                             #   unk = np.random.uniform(-0.1, 0.1, settings.WORD_EMBEDDING_SIZE).tolist()  #####
                                sentence_c.append(unk)
                        for _ in range(max_S_len - len(words)):
                            sentence_c.append(padding)
                        context_test.append(sentence_c)
            w.close()

    context_train = np.array(context_train, dtype=float)
    target_train = np.array(target_train, dtype=float)
    data_train = np.concatenate((context_train, target_train), axis=1)
    # concatenate the respective target (max_S_len ~ max_S_len + max_T_len -1) after the context (0 ~ max_S_len - 1)
    label_train = np.array(label_train, dtype=float)

    context_test = np.array(context_test, dtype=float)
    target_test = np.array(target_test, dtype=float)
    data_test = np.concatenate((context_test, target_test), axis=1)
    # concatenate the respective target (max_S_len ~ max_S_len + max_T_len -1) after the context (0 ~ max_S_len - 1)
    label_test = np.array(label_test, dtype=float)

    return data_train, label_train, data_test, label_test


if __name__ == "__main__":
    control_dict = {'laptops': 1, 'restaurant': 0}
    # by setting the value of 'laptops' and 'restaurant' to 0/1, control the generation of datasets
    _, _, word_dict, max_S_len, max_T_len = create_vocabulary(laptop=control_dict['laptops'],
                                                              restaurant=control_dict['restaurant'])
    word_embedding_dict = create_word_embedding_dict(word_dict)
    data_train, label_train, data_test, label_test = make_dataset(word_embedding_dict, max_S_len, max_T_len,
                                                                  laptop=control_dict['laptops'],
                                                                  restaurant=control_dict['restaurant'])

    if control_dict['laptops'] == 1 and control_dict['restaurant'] == 1:
        path_data_train = 'data_processed/train_data_all_vocab' + str(settings.VOCAB_SIZE) + '_' + str(
            settings.WORD_EMBEDDING_SIZE) + '.npy '
        path_label_train = 'data_processed/train_label_all_vocab' + str(settings.VOCAB_SIZE) + '_' + str(
            settings.WORD_EMBEDDING_SIZE) + '.npy'
        path_data_test = 'data_processed/test_data_all_vocab' + str(settings.VOCAB_SIZE) + '_' + str(
            settings.WORD_EMBEDDING_SIZE) + '.npy '
        path_label_test = 'data_processed/test_label_all_vocab' + str(settings.VOCAB_SIZE) + '_' + str(
            settings.WORD_EMBEDDING_SIZE) + '.npy'

    if control_dict['laptops'] == 1 and control_dict['restaurant'] == 0:
        path_data_train = 'data_processed/train_data_laptop_vocab' + str(settings.VOCAB_SIZE) + '_' + str(
            settings.WORD_EMBEDDING_SIZE) + '.npy'
        path_label_train = 'data_processed/train_label_laptop_vocab' + str(settings.VOCAB_SIZE) + '_' + str(
            settings.WORD_EMBEDDING_SIZE) + '.npy'
        path_data_test = 'data_processed/test_data_laptop_vocab' + str(settings.VOCAB_SIZE) + '_' + str(
            settings.WORD_EMBEDDING_SIZE) + '.npy'
        path_label_test = 'data_processed/test_label_laptop_vocab' + str(settings.VOCAB_SIZE) + '_' + str(
            settings.WORD_EMBEDDING_SIZE) + '.npy'
    if control_dict['laptops'] == 0 and control_dict['restaurant'] == 1:
        path_data_train = 'data_processed/train_data_restaurant_vocab' + str(settings.VOCAB_SIZE) + '_' + str(
            settings.WORD_EMBEDDING_SIZE) + '.npy'
        path_label_train = 'data_processed/train_label_restaurant_vocab' + str(settings.VOCAB_SIZE) + '_' + str(
            settings.WORD_EMBEDDING_SIZE) + '.npy'
        path_data_test = 'data_processed/test_data_restaurant_vocab' + str(settings.VOCAB_SIZE) + '_' + str(
            settings.WORD_EMBEDDING_SIZE) + '.npy'
        path_label_test = 'data_processed/test_label_restaurant_vocab' + str(settings.VOCAB_SIZE) + '_' + str(
            settings.WORD_EMBEDDING_SIZE) + '.npy'

    np.save(path_data_train, data_train)
    np.save(path_label_train, label_train)
    np.save(path_data_test, data_test)
    np.save(path_label_test, label_test)

    print('data_train shape {}'.format(data_train.shape))
    print('label_train shape {}'.format(label_train.shape))
    print('data_test shape {}'.format(data_test.shape))
    print('label_test shape {}'.format(label_test.shape))
    print('max context length: {}, max target length: {}'.format(max_S_len, max_T_len))
