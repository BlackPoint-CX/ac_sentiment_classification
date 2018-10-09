#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : imbd.py
__create_time__ : 2018/10/09
"""

import tensorflow as tf

from tensorflow import keras

import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()

word_index = {k: v + 3 for k, v in word_index.items()}  # 空出来前三位方便加其他元素 word_index 从1开始计数
word_index['<PAD>'] = 0
word_index['<UNK>'] = 1
word_index['<UNUSED>'] = 2

reversed_word_index = {v: k for k, v in word_index.items()}


def decode_review(text):
    return ' '.join([reversed_word_index.get(x, '?') for x in text])


train_data = pad_sequences(train_data, value=word_index['<PAD>'], padding='post', maxlen=256)

test_data = pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxlen=256)

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(units=16, activation=tf.nn.relu))
model.add(keras.layers.Dense(units=1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='binary_crossentropy', metrics=['accuracy'])

x_val = train_data[:10000]
x_train = train_data[10000:]
y_val = train_labels[:10000]
y_train = train_labels[10000:]

histroy = model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose=1)
