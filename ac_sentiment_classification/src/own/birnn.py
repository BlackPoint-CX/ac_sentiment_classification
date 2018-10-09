#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : baseline_model.py
__create_time__ : 2018/10/08
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib.rnn import BasicRNNCell, LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
import numpy as np


class BaselineModel():
    def __init__(self, config):
        self.config = config
        self.train_graph = tf.Graph()
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def build_train_graph(self):
        self.init_train_placeholder()

    def init_train_placeholder(self):
        with self.train_graph.as_default(), tf.name_scope('init_train_placeholder'):
            self.input_token_ids = tf.placeholder(dtype=tf.float32, shape=[None, self.config.token_max_len],
                                                  name='input_placeholder')
            self.input_seqs_len = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_seq_len')
            self.input_label_ids = tf.placeholder(dtype=tf.int32, shape=[None, 4])

    def init_train_embedding(self):
        with self.train_graph.as_default(), tf.name_scope('init_train_embedding'):
            if self.config.pretrain_embedding:
                self.embeddings = tf.Variable(initial_value=self.config.pretrain_embedding, name='embedding')
            else:
                self.embeddings = tf.get_variable(name='embedding',
                                                  shape=[self.config.n_token, self.config.embedding_dim],
                                                  dtype=tf.float32, initializer=tf.random_normal_initializer())

    def init_lookup(self):
        with self.train_graph.as_default(), tf.name_scope('init_lookup'):
            self.token_embeddings = tf.nn.embedding_lookup(params=self.embeddings, ids=self.input_token_ids,
                                                           name='token_embedding')

    def init_train(self):
        with self.train_graph.as_default():
            with tf.name_scope('init_train_birnn'):
                cell_fw = LSTMCell(num_units=self.config.n_hidden_units)
                cell_bw = BasicRNNCell(num_units=self.config.n_hidden_units)
                ((output_fw, output_bw), (output_state_fw, output_state_bw)) = bidirectional_dynamic_rnn(
                    cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.token_embeddings, sequence_length=self.input_seqs_len)

                self.bi_rnn_output = tf.concat([output_fw, output_bw], axis=1, name='bi_rnn_output')

            with tf.name_scope('init_train_proj'):
                W = tf.get_variable(dtype=tf.float32, shape=[2 * self.config.n_hidden_units, self.config.n_labels],
                                    name='proj_weight')
                b = tf.get_variable(dtype=tf.float32, shape=[self.config.n_labels], name='proj_bias')

                self.train_logits = tf.nn.softmax(tf.matmul(self.bi_rnn_output, W) + b)  # batch_size * n_labels

    def init_train_loss(self):
        with self.train_graph.as_default():
            with tf.name_scope('init_train_loss'):
                self.train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(lables=self.input_label_ids,
                                                                                 logits=self.train_logits)

    def init_predict(self):
        with self.train_graph.as_default():
            with tf.name_scope('init_predict'):
                self.predicts = tf.argmax(self.train_logits, axis=-1)

    def init_train_eval(self, val):
        with self.train_graph.as_default():
            with tf.name_scope('init_train_eval'):
                accuracy = 0
                n_total = len(val)
                for token_ids, labels in val:
                    predict = self.sess.run([self.predicts], feed_dict={self.input_token_ids: token_ids})
                    labels = np.argmax(labels, axis=1)
                    for single_predict, single_label in zip(predict, labels):
                        if single_predict == single_label:
                            accuracy += 1
                accuracy /= n_total

        return accuracy







