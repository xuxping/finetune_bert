# -*- coding:utf-8 -*-
# Date: 2019/9/26 15:09
# Author: xuxiaoping
# Desc: Keras bert model for sequence classification

from keras_bert import load_trained_model_from_checkpoint

from tensorflow.python.keras.layers import Dense, Lambda, Input
from tensorflow.python.keras.models import Model


def BertForSequenceClassification(config):
    bert_model = load_trained_model_from_checkpoint(config.bert_config_path,
                                                    config.bert_checkpoint_path,
                                                    seq_len=None)

    for l in bert_model.layers:
        l.trainable = True
    x1_in = Input(shape=(None,))  # token ids input
    x2_in = Input(shape=(None,))  # segment ids input

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)  # get first token embedding
    p = Dense(2, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)

    return model
