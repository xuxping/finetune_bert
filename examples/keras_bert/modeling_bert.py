# -*- coding:utf-8 -*-
# Desc: Keras bert model for sequence classification

from keras_bert import load_trained_model_from_checkpoint
import os
from tensorflow.python.keras.layers import Dense, Lambda, Input, Dropout
from tensorflow.python.keras.models import Model


class BertForSequenceClassification():

    @staticmethod
    def build(config):
        bert_config_path = os.path.join(config.pretrained_path, 'bert_config.json')
        bert_checkpoint_path = os.path.join(config.pretrained_path, 'bert_model.ckpt')
        bert_model = load_trained_model_from_checkpoint(bert_config_path,
                                                        bert_checkpoint_path,
                                                        seq_len=None)

        for l in bert_model.layers:
            l.trainable = True
        x1_in = Input(shape=(None,))  # token ids input
        x2_in = Input(shape=(None,))  # segment ids input

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)  # get first token embedding
        x = Dropout(config.hidden_dropout_prob)(x)
        p = Dense(2, activation='softmax')(x)

        model = Model([x1_in, x2_in], p)

        return model
