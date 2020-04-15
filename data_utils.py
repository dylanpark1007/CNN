import numpy as np
import os
import re
from math import ceil
import codecs

MR_len = 54
Subj_len = 115
Trec_len = 54
MPQA_len = 35
SST1_len = 100
SST2_len = 54
CR_len = 100


def clean_string(string_list):
    ret_list = []
    for string in string_list:
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = re.sub(r"[\.,-]", "", string)
        ret_list.append(string)
    return ret_list


def build_word_index(string_list):
    string_list = clean_string(string_list)
    word2idx = {}
    for line in string_list:
        for word in line.split():
            if not word in word2idx:
                word2idx[word] = len(word2idx) + 1
    return word2idx


def tokenizer(string_list, padding, word2idx):
    string_list = clean_string(string_list)
    tokenized = []
    for line in string_list:
        tokenized_line = []
        for word in line.split():
            tokenized_line.append(word2idx[word])
        k = padding - len(tokenized_line)
        tokenized_line += [0] * k
        tokenized.append(tokenized_line)
    return np.asarray(tokenized)


def get_data(dataset):
    if dataset == 'MR':
        paths = ['data/rt-polarity.pos', 'data/rt-polarity.neg']
        PATH_POS = paths[0]
        PATH_NEG = paths[1]
        with codecs.open(PATH_NEG, 'r', encoding='utf-8', errors='ignore') as f:
            neg_texts = f.read().splitlines()
        with codecs.open(PATH_POS, 'r', encoding='utf-8', errors='ignore') as f:
            pos_texts = f.read().splitlines()

        word2idx = build_word_index(string_list=(
                clean_string(pos_texts) + clean_string(neg_texts)))
        t_pos = tokenizer(pos_texts, MR_len, word2idx)
        t_neg = tokenizer(neg_texts, MR_len, word2idx)

        pos_labels = np.ones([t_pos.shape[0], ], dtype='int32')
        neg_labels = np.zeros([t_neg.shape[0], ], dtype='int32')
        data = np.concatenate((t_pos, t_neg))

        labels = np.concatenate((pos_labels, neg_labels))


    elif dataset == 'Subj':
        paths = ['data/sentiment_dataset-master/sentiment_dataset-master/data/subj.all']
        PATH = paths[0]
        labels = []
        texts = []
        with codecs.open(PATH, 'r', encoding='utf-8', errors='ignore') as f:
            corpus = f.read().splitlines()

        for line in corpus:
            line_token = line.split()
            labels.append(line_token[0])
            sentence = ' '.join(line_token[1::])
            texts.append(sentence)

        word2idx = build_word_index(string_list=(clean_string(texts)))
        data = tokenizer(texts, Subj_len, word2idx)
        labels = np.array(labels, dtype='int32')
        maxval= 0
        for i in data:
            if len(i) > maxval:
                maxval = len(i)
        print(maxval)


    elif dataset == 'Trec' :
        paths = ['data/sentiment_dataset-master/sentiment_dataset-master/data/TREC2_train.txt','data/sentiment_dataset-master/sentiment_dataset-master/data/TREC2_test.txt']
        PATH_train = paths[0]
        PATH_test = paths[1]
        labels = []
        texts = []
        with codecs.open(PATH_train, 'r', encoding='utf-8', errors='ignore') as f:
            corpus_train = f.read().splitlines()
        with codecs.open(PATH_test, 'r', encoding='utf-8', errors='ignore') as f:
            corpus_test = f.read().splitlines()


        for line in corpus_train:
            line_token = line.split()
            label_token = line_token[0].split(':')[0]
            if label_token == 'DESC':
                label_token = 0
            elif label_token == 'ENTY':
                label_token = 1
            elif label_token == 'ABBR':
                label_token = 2
            elif label_token == 'HUM':
                label_token = 3
            elif label_token == 'LOC':
                label_token = 4
            elif label_token == 'NUM':
                label_token = 5
            labels.append(label_token)
            sentence = ' '.join(line_token[1::])
            texts.append(sentence)
        for line in corpus_test:
            line_token = line.split()
            label_token = line_token[0].split(':')[0]
            if label_token == 'DESC':
                label_token = 0
            elif label_token == 'ENTY':
                label_token = 1
            elif label_token == 'ABBR':
                label_token = 2
            elif label_token == 'HUM':
                label_token = 3
            elif label_token == 'LOC':
                label_token = 4
            elif label_token == 'NUM':
                label_token = 5
            labels.append(label_token)
            sentence = ' '.join(line_token[1::])
            texts.append(sentence)

        word2idx = build_word_index(string_list=(clean_string(texts)))
        data = tokenizer(texts, Trec_len, word2idx)
        labels = np.array(labels, dtype='int32')


    elif dataset == 'MPQA':
        paths = ['data/sentiment_dataset-master/sentiment_dataset-master/data/mpqa.all']
        PATH = paths[0]
        labels = []
        texts = []
        with codecs.open(PATH, 'r', encoding='utf-8', errors='ignore') as f:
            corpus = f.read().splitlines()

        for line in corpus:
            line_token = line.split()
            labels.append(line_token[0])
            sentence = ' '.join(line_token[1::])
            texts.append(sentence)

        word2idx = build_word_index(string_list=(clean_string(texts)))
        data = tokenizer(texts, MPQA_len, word2idx)
        labels = np.array(labels, dtype='int32')


    elif dataset == 'SST-2' :
        paths = ['data/sentiment_dataset-master/sentiment_dataset-master/data/stsa.binary.phrases.train',
                 'data/sentiment_dataset-master/sentiment_dataset-master/data/stsa.binary.dev',
                 'data/sentiment_dataset-master/sentiment_dataset-master/data/stsa.binary.test',
                 'data/sentiment_dataset-master/sentiment_dataset-master/data/stsa.binary.train']
        PATH_phrase = paths[0]
        PATH_dev = paths[1]
        PATH_test = paths[2]
        PATH_train = paths[3]
        labels = []
        texts = []
        with codecs.open(PATH_phrase, 'r', encoding='utf-8', errors='ignore') as f:
            corpus_phrase = f.read().splitlines()
        with codecs.open(PATH_dev, 'r', encoding='utf-8', errors='ignore') as f:
            corpus_dev = f.read().splitlines()
        with codecs.open(PATH_test, 'r', encoding='utf-8', errors='ignore') as f:
            corpus_test = f.read().splitlines()
        with codecs.open(PATH_train, 'r', encoding='utf-8', errors='ignore') as f:
            corpus_train = f.read().splitlines()


        for line in corpus_phrase:
            line_token = line.split()
            labels.append(line_token[0])
            sentence = ' '.join(line_token[1::])
            texts.append(sentence)
        for line in corpus_dev:
            line_token = line.split()
            labels.append(line_token[0])
            sentence = ' '.join(line_token[1::])
            texts.append(sentence)
        for line in corpus_test:
            line_token = line.split()
            labels.append(line_token[0])
            sentence = ' '.join(line_token[1::])
            texts.append(sentence)
        for line in corpus_train:
            line_token = line.split()
            labels.append(line_token[0])
            sentence = ' '.join(line_token[1::])
            texts.append(sentence)

        word2idx = build_word_index(string_list=(clean_string(texts)))
        data = tokenizer(texts, SST2_len, word2idx)
        labels = np.array(labels, dtype='int32')


    elif dataset == 'SST-1' :
        paths = ['data/sentiment_dataset-master/sentiment_dataset-master/data/stsa.fine.phrases.train',
                 'data/sentiment_dataset-master/sentiment_dataset-master/data/stsa.fine.dev',
                 'data/sentiment_dataset-master/sentiment_dataset-master/data/stsa.fine.test',
                 'data/sentiment_dataset-master/sentiment_dataset-master/data/stsa.fine.train']
        PATH_phrase = paths[0]
        PATH_dev = paths[1]
        PATH_test = paths[2]
        PATH_train = paths[3]
        labels = []
        texts = []
        with codecs.open(PATH_phrase, 'r', encoding='utf-8', errors='ignore') as f:
            corpus_phrase = f.read().splitlines()
        with codecs.open(PATH_dev, 'r', encoding='utf-8', errors='ignore') as f:
            corpus_dev = f.read().splitlines()
        with codecs.open(PATH_test, 'r', encoding='utf-8', errors='ignore') as f:
            corpus_test = f.read().splitlines()
        with codecs.open(PATH_train, 'r', encoding='utf-8', errors='ignore') as f:
            corpus_train = f.read().splitlines()


        for line in corpus_phrase:
            line_token = line.split()
            labels.append(line_token[0])
            sentence = ' '.join(line_token[1::])
            texts.append(sentence)
        for line in corpus_dev:
            line_token = line.split()
            labels.append(line_token[0])
            sentence = ' '.join(line_token[1::])
            texts.append(sentence)
        for line in corpus_test:
            line_token = line.split()
            labels.append(line_token[0])
            sentence = ' '.join(line_token[1::])
            texts.append(sentence)
        for line in corpus_train:
            line_token = line.split()
            labels.append(line_token[0])
            sentence = ' '.join(line_token[1::])
            texts.append(sentence)

        word2idx = build_word_index(string_list=(clean_string(texts)))
        data = tokenizer(texts, SST1_len, word2idx)
        labels = np.array(labels, dtype='int32')



    elif dataset == 'CR' :
        paths = ['data/sentiment_dataset-master/sentiment_dataset-master/data/CR_negative-words.txt',
                 'data/sentiment_dataset-master/sentiment_dataset-master/data/CR_positive-words.txt',
                 'data/sentiment_dataset-master/sentiment_dataset-master/data/custrev.all']
        PATH_neg_word = paths[0]
        PATH_pos_word = paths[1]
        PATH_sentence = paths[2]
        labels = []
        texts = []
        with codecs.open(PATH_neg_word, 'r', encoding='utf-8', errors='ignore') as f:
            corpus_neg_word = f.read().splitlines()
        with codecs.open(PATH_pos_word, 'r', encoding='utf-8', errors='ignore') as f:
            corpus_pos_word = f.read().splitlines()
        with codecs.open(PATH_sentence, 'r', encoding='utf-8', errors='ignore') as f:
            corpus_sentence = f.read().splitlines()

        for line in corpus_neg_word:
            if ';' in line:
                continue
            line_token = line.split()
            labels.append(0)
            sentence = ' '.join(line_token[::])
            texts.append(sentence)
        for line in corpus_pos_word:
            if ';' in line:
                continue
            line_token = line.split()
            labels.append(1)
            sentence = ' '.join(line_token[::])
            texts.append(sentence)
        for line in corpus_sentence:
            line_token = line.split()
            labels.append(line_token[0])
            sentence = ' '.join(line_token[1::])
            texts.append(sentence)

        word2idx = build_word_index(string_list=(clean_string(texts)))
        data = tokenizer(texts, CR_len, word2idx)
        labels = np.array(labels, dtype='int32')


    return data, labels, word2idx



def generate_split(data, labels, val_split):
    j = np.concatenate((data, labels.reshape([-1, 1])), 1)
    np.random.shuffle(j)
    split_point = int(ceil(data.shape[0]*(1-val_split)))
    train_data = j[:split_point,:-1]
    val_data = j[split_point:,:-1]
    train_labels = j[:split_point,-1]
    val_labels = j[split_point:, -1]
    return train_data, train_labels, val_data, val_labels

def generate_batch(data, labels, batch_size):
    j = np.concatenate((data, labels.reshape([-1, 1])), 1)
    mark = np.random.randint(batch_size, j.shape[0])
    batch_data = j[mark-batch_size : mark]
    return batch_data[:,:-1], batch_data[:,-1]

