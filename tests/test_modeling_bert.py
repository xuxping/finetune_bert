import os
import sys

print(sys.path)
sys.path.append("/home/xuxiaoping/nlp/finetune_bert")
from finetune.configuration_bert import BertConfig
from finetune.modeling_bert import BertForSequenceClassification
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

BERT_PRETRAINED_PATH = "/data/xuxiaoping/npai/pretrained/bert/chinese_L-12_H-768_A-12/"


def test_classification_input():
    pretrained_path = "/data/xuxiaoping/npai/pretrained/bert/chinese_L-12_H-768_A-12"

    config_path = os.path.join(pretrained_path, "bert_config.json")
    config = BertConfig.from_pretrained(config_path)
    config.num_labels = 2

    model = BertForSequenceClassification.build(config, pretrained_path)
    token_inputs = tf.constant([[7, 6, 0, 0, 0]], dtype=tf.int32)
    # position_inputs = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)
    token_type_inputs = tf.constant([[0, 0, 0, 0, 0]], dtype=tf.int32)

    ret = model([token_inputs, token_type_inputs],
                training=False)  # build the network with dummy inputs
    print(ret)
    model.summary()
    print(config.to_dict())


class Config(object):
    bert_config_path = os.path.join(BERT_PRETRAINED_PATH, "bert_config.json")
    bert_checkpoint_path = os.path.join(BERT_PRETRAINED_PATH, "bert_model.ckpt")
    num_labels = 2
    hidden_dropout_prob = 0.1
    initializer_range=0.02


def test_classification():
    config = Config()

    model = BertForSequenceClassification.build(config)
    model.summary()
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(1e-5),
        metrics=['accuracy'])  # 用足够小的学习率


if __name__ == '__main__':
    test_classification()
