import os
import sys
import random
import tensorflow as tf
import unittest

sys.path.append("../")
from finetune import (DistillBertConfig, DistillBertTokenizer, DistillBertForPretraining,
                      DistillBertForSequenceClassification)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
DistillBERT_PRETRAINED_PATH = "../configs/distillbert"
DistillBERT_VOCAB_PATH = os.path.join(DistillBERT_PRETRAINED_PATH, 'vocab.txt')
DistillBERT_CONFIG_PATH = os.path.join(DistillBERT_PRETRAINED_PATH, "config.json")


class TestDistillBertModel(unittest.TestCase):

    def setUp(self):
        self.tokenizer = DistillBertTokenizer.from_pretrained(DistillBERT_VOCAB_PATH)
        self.config = DistillBertConfig.from_pretrained(DistillBERT_CONFIG_PATH)

    def test_encode(self):
        text_a = "今天天气很不错噢"
        token_ids, segment_ids, input_mask = self.tokenizer.encode(text_a=text_a, max_seq_length=128)
        print(token_ids)
        print(segment_ids)
        print(input_mask)

    def test_pair_encode(self):
        text_a = "今天天气很不错噢"
        text_b = "明天天气也是很不错的噢"
        token_ids, segment_ids, input_mask = self.tokenizer.encode(text_a=text_a, text_b=text_b, max_seq_length=128)
        print(token_ids)
        print(segment_ids)
        print(input_mask)

    def test_pretraining_model(self):
        # bert = DistillBertForPretraining(self.config, training=True, trainable=True)
        # bert.build()
        bert = DistillBertForPretraining.from_pretrained(pretrained_path=DistillBERT_PRETRAINED_PATH,
                                                         training=True)

        bert.model.summary()

    def test_for_sequence_classification_model(self):
        bert = DistillBertForSequenceClassification(self.config, training=True, trainable=True)
        bert.build()
        # bert = DistillBertForSequenceClassification.from_pretrained(pretrained_path=DistillBERT_PRETRAINED_PATH,
        #                                                             training=True,
        #                                                             num_labels=2)

        bert.model.summary()
        input_ids = tf.constant([[7, 6, 0, 0, 0]], dtype=tf.int32)
        token_type_ids = tf.constant([[0, 0, 0, 0, 0]], dtype=tf.int32)
        ret = bert.model([input_ids, token_type_ids])  # build the network with dummy inputs
        print(ret)


if __name__ == '__main__':
    # python -m unittest test_modeling_bert.TestBertModel.test_encode
    unittest.main()
