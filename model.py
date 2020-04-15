import tensorflow as tf
import numpy as np
from math import ceil
import data_utils
import sys
import configuration
from gensim.models import KeyedVectors



class CNN(object):

    def __init__(self, config, sess, w2idx):
        self.n_epochs = config['n_epochs']
        self.kernel_sizes = config['kernel_sizes']
        self.n_filters = config['n_filters']
        self.dropout_rate = config['dropout_rate']
        self.val_split = config['val_split']
        self.edim = config['edim']
        self.n_words = config['n_words']
        self.std_dev = config['std_dev']
        self.input_len = config['sentence_len']
        self.batch_size = config['batch_size']
        self.inp = tf.placeholder(shape=[None, self.input_len], dtype='int32')
        self.labels = tf.placeholder(shape=[None, ], dtype='int32')
        self.loss = None
        self.session = sess
        self.cur_drop_rate = tf.placeholder(dtype='float32')
        self.w2idx = w2idx
        self.l2_loss = tf.constant(0.0)

    def build_model(self):
        if configuration.config['model option'] == 'rand':
            word_embedding = tf.Variable(tf.random_normal([self.n_words, self.edim], stddev=self.std_dev))


        elif configuration.config['model option'] == 'static':
            model = KeyedVectors.load_word2vec_format(
                r'C:\Users\dilab\PycharmProjects\doc2vec\data\GoogleNews-vectors-negative300.bin.gz', binary=True)

            reversed_word2idx={}
            pre_vector = []
            for k, v in self.w2idx.items():
                reversed_word2idx[v] = k
            temp = np.random.uniform(low=-0.5, high=0.5, size=[300, ])
            pre_vector.append(temp)
            for index in reversed_word2idx:
                if reversed_word2idx[index] not in model:
                    temp = np.random.uniform(low = -0.5,high=0.5,size=[300,])
                    pre_vector.append(temp)
                    continue
                pre_vector.append(model[reversed_word2idx[index]])
            pre_vector = np.array(pre_vector)
            pre_vector = tf.convert_to_tensor(pre_vector, np.float32)
            word_embedding = tf.Variable(pre_vector, trainable=False)

        elif configuration.config['model option'] == 'non-static':
            model = KeyedVectors.load_word2vec_format(
                r'C:\Users\dilab\PycharmProjects\doc2vec\data\GoogleNews-vectors-negative300.bin.gz', binary=True)

            reversed_word2idx={}
            pre_vector = []
            for k, v in self.w2idx.items():
                reversed_word2idx[v] = k
            temp = np.random.uniform(low=-0.5, high=0.5, size=[300, ])
            pre_vector.append(temp)
            for index in reversed_word2idx:
                if reversed_word2idx[index] not in model:
                    temp = np.random.uniform(low = -0.5,high=0.5,size=[300,])
                    pre_vector.append(temp)
                    continue
                pre_vector.append(model[reversed_word2idx[index]])
            pre_vector = np.array(pre_vector)
            pre_vector = tf.convert_to_tensor(pre_vector, np.float32)
            word_embedding = tf.Variable(pre_vector)

        elif configuration.config['model option'] == 'multichannel':
            model = KeyedVectors.load_word2vec_format(
                r'C:\Users\dilab\PycharmProjects\doc2vec\data\GoogleNews-vectors-negative300.bin.gz', binary=True)

            reversed_word2idx={}
            pre_vector = []
            for k, v in self.w2idx.items():
                reversed_word2idx[v] = k
            temp = np.random.uniform(low=-0.5, high=0.5, size=[300, ])
            pre_vector.append(temp)
            for index in reversed_word2idx:
                if reversed_word2idx[index] not in model:
                    temp = np.random.uniform(low = -0.5,high=0.5,size=[300,])
                    pre_vector.append(temp)
                    continue
                pre_vector.append(model[reversed_word2idx[index]])
            pre_vector = np.array(pre_vector)
            pre_vector = tf.convert_to_tensor(pre_vector, np.float32)
            word_embedding = tf.Variable(pre_vector)
            word_embedding2 = tf.Variable(pre_vector,trainable=False)


        else:
            print('mode type error')
            exit()



        if configuration.config['model option'] == 'multichannel':
            x = tf.nn.embedding_lookup(word_embedding, self.inp)
            x2 = tf.nn.embedding_lookup(word_embedding2, self.inp)
            x_conv = tf.expand_dims(x, -1)
            x_conv2 = tf.expand_dims(x2, -1)

            # Filters
            F1 = tf.Variable(
                tf.random_normal([self.kernel_sizes[0], self.edim, 1, self.n_filters], stddev=self.std_dev),
                dtype='float32')
            F2 = tf.Variable(
                tf.random_normal([self.kernel_sizes[1], self.edim, 1, self.n_filters], stddev=self.std_dev),
                dtype='float32')
            F3 = tf.Variable(
                tf.random_normal([self.kernel_sizes[2], self.edim, 1, self.n_filters], stddev=self.std_dev),
                dtype='float32')
            FB1 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
            FB2 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
            FB3 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
            # Weight for final layer
            if configuration.config['dataset'] == 'Trec':
                num_class = 6
            elif configuration.config['dataset'] == 'SST-1':
                num_class = 5
            else:
                num_class = 2
            W = tf.Variable(tf.random_normal([6 * self.n_filters, num_class], stddev=self.std_dev), dtype='float32')
            b = tf.Variable(tf.constant(0.1, shape=[1, num_class]), dtype='float32')
            # Convolutions
            C1 = tf.add(tf.nn.conv2d(x_conv, F1, [1, 1, 1, 1], padding='VALID'), FB1)
            C2 = tf.add(tf.nn.conv2d(x_conv, F2, [1, 1, 1, 1], padding='VALID'), FB2)
            C3 = tf.add(tf.nn.conv2d(x_conv, F3, [1, 1, 1, 1], padding='VALID'), FB3)
            C4 = tf.add(tf.nn.conv2d(x_conv2, F1, [1, 1, 1, 1], padding='VALID'), FB1)
            C5 = tf.add(tf.nn.conv2d(x_conv2, F2, [1, 1, 1, 1], padding='VALID'), FB2)
            C6 = tf.add(tf.nn.conv2d(x_conv2, F3, [1, 1, 1, 1], padding='VALID'), FB3)

            C1 = tf.nn.relu(C1)
            C2 = tf.nn.relu(C2)
            C3 = tf.nn.relu(C3)
            C4 = tf.nn.relu(C4)
            C5 = tf.nn.relu(C5)
            C6 = tf.nn.relu(C6)

            # Max pooling
            maxC1 = tf.nn.max_pool(C1, [1, C1.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
            maxC1 = tf.squeeze(maxC1, [1, 2])
            maxC2 = tf.nn.max_pool(C2, [1, C2.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
            maxC2 = tf.squeeze(maxC2, [1, 2])
            maxC3 = tf.nn.max_pool(C3, [1, C3.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
            maxC3 = tf.squeeze(maxC3, [1, 2])
            maxC4 = tf.nn.max_pool(C4, [1, C1.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
            maxC4 = tf.squeeze(maxC4, [1, 2])
            maxC5 = tf.nn.max_pool(C5, [1, C2.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
            maxC5 = tf.squeeze(maxC5, [1, 2])
            maxC6 = tf.nn.max_pool(C6, [1, C3.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
            maxC6 = tf.squeeze(maxC6, [1, 2])
            # Concatenating pooled features
            z = tf.concat(axis=1, values=[maxC1, maxC2, maxC3, maxC4, maxC5, maxC6])
            zd = tf.nn.dropout(z, self.cur_drop_rate)
            # Fully connected layer
            self.y = tf.add(tf.matmul(zd, W), b)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.labels)

            self.loss = tf.reduce_mean(losses)
            self.optim = tf.train.AdamOptimizer(learning_rate=0.001)
            self.train_op = self.optim.minimize(self.loss)

        else:
            x = tf.nn.embedding_lookup(word_embedding, self.inp)
            x_conv = tf.expand_dims(x, -1)

            # Filters
            F1 = tf.Variable(
                tf.random_normal([self.kernel_sizes[0], self.edim, 1, self.n_filters], stddev=self.std_dev),
                dtype='float32')
            F2 = tf.Variable(
                tf.random_normal([self.kernel_sizes[1], self.edim, 1, self.n_filters], stddev=self.std_dev),
                dtype='float32')
            F3 = tf.Variable(
                tf.random_normal([self.kernel_sizes[2], self.edim, 1, self.n_filters], stddev=self.std_dev),
                dtype='float32')
            FB1 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
            FB2 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
            FB3 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
            # Weight for final layer
            if configuration.config['dataset'] == 'Trec':
                num_class = 6
            elif configuration.config['dataset'] == 'SST-1':
                num_class = 5
            else:
                num_class = 2
            W = tf.Variable(tf.random_normal([3 * self.n_filters, num_class], stddev=self.std_dev), dtype='float32')
            b = tf.Variable(tf.constant(0.1, shape=[1, num_class]), dtype='float32')
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            # Convolutions
            C1 = tf.add(tf.nn.conv2d(x_conv, F1, [1, 1, 1, 1], padding='VALID'), FB1)
            C2 = tf.add(tf.nn.conv2d(x_conv, F2, [1, 1, 1, 1], padding='VALID'), FB2)
            C3 = tf.add(tf.nn.conv2d(x_conv, F3, [1, 1, 1, 1], padding='VALID'), FB3)

            C1 = tf.nn.relu(C1)
            C2 = tf.nn.relu(C2)
            C3 = tf.nn.relu(C3)

            # Max pooling
            maxC1 = tf.nn.max_pool(C1, [1, C1.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
            maxC1 = tf.squeeze(maxC1, [1, 2])
            maxC2 = tf.nn.max_pool(C2, [1, C2.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
            maxC2 = tf.squeeze(maxC2, [1, 2])
            maxC3 = tf.nn.max_pool(C3, [1, C3.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
            maxC3 = tf.squeeze(maxC3, [1, 2])
            # Concatenating pooled features
            z = tf.concat(axis=1, values=[maxC1, maxC2, maxC3])
            zd = tf.nn.dropout(z, self.cur_drop_rate)
            # Fully connected layer
            self.y = tf.add(tf.matmul(zd, W), b)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.labels)

            # self.loss = tf.reduce_mean(losses) + 0.01*tf.nn.l2_loss(C1) + 0.01*tf.nn.l2_loss(C2) + 0.01*tf.nn.l2_loss(C3) # version2

            l2_reg_lambda = 0.1 # version1 41%
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2_loss

            # self.loss = tf.reduce_mean(losses) # original version
            self.optim = tf.train.AdamOptimizer(learning_rate=0.001)
            self.train_op = self.optim.minimize(self.loss)




    def train(self, data, labels):
        self.build_model()
        if configuration.config['dataset'] == 'CR':
            data = data[6792::]
            labels = labels[6792::]
            train_data = data[0:6791]
            train_labels = labels[0:6791]
        n_batches = int(ceil(data.shape[0] / self.batch_size))
        tf.global_variables_initializer().run()
        t_data, t_labels, v_data, v_labels = data_utils.generate_split(data, labels, self.val_split)
        if configuration.config['dataset'] == 'CR':
            d1, l1, d2, l2 = data_utils.generate_split(train_data, train_labels, self.val_split)
            train_data = np.concatenate([d1, d2],0)
            train_labels = np.concatenate([l1, l2],0)
            t_data = np.concatenate([t_data, train_data],0)
            t_labels = np.concatenate([t_labels, train_labels],0)
        for epoch in range(1, self.n_epochs + 1):
            train_cost = 0
            for batch in range(1, n_batches + 1):
                X, y = data_utils.generate_batch(t_data, t_labels, self.batch_size)
                f_dict = {
                    self.inp: X,
                    self.labels: y,
                    self.cur_drop_rate: self.dropout_rate
                }

                _, cost = self.session.run([self.train_op, self.loss], feed_dict=f_dict)
                train_cost += cost
                sys.stdout.write('Epoch %d Cost  :   %f - Batch %d of %d     \r' % (epoch, cost, batch, n_batches))
                sys.stdout.flush()

            print(self.test(v_data, v_labels))

    def test(self, data, labels):
        n_batches = int(ceil(data.shape[0] / self.batch_size))
        test_cost = 0
        preds = []
        ys = []
        for batch in range(1, n_batches + 1):
            X, Y = data_utils.generate_batch(data, labels, self.batch_size)
            f_dict = {
                self.inp: X,
                self.labels: Y,
                self.cur_drop_rate: 1.0
            }
            cost, y = self.session.run([self.loss, self.y], feed_dict=f_dict)
            test_cost += cost
            sys.stdout.write('Cost  :   %f - Batch %d of %d     \r' % (cost, batch, n_batches))
            sys.stdout.flush()

            preds.extend(np.argmax(y, 1))
            ys.extend(Y)


        print("Accuracy", np.mean(np.asarray(np.equal(ys, preds), dtype='float32')) * 100)


# 1. 224줄에 L2 norm 구현


import math
def sigmoid_table():
    EXP_TABLE_SIZE = 1000
    MAX_EXP = 6
    expTable = []
    for i in range(EXP_TABLE_SIZE):
        expvalue = math.exp((i/EXP_TABLE_SIZE*2-1)*MAX_EXP)
        expTable.append(expvalue / (expvalue+1))
    return expTable

t = sigmoid_table()