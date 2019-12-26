import os
import sys
import random
import tensorflow as tf
import unittest

print(sys.path)
sys.path.append("/home/xuxiaoping/nlp/finetune_bert")
from finetune.configuration_bert import BertConfig
from finetune.tokenization_bert import BertTokenizer
from finetune.modeling_bert import BertForPretraining, BertForSequenceClassification

os.environ["CUDA_VISIBLE_DEVICES"] = ""
BERT_PRETRAINED_PATH = "/data/xuxiaoping/npai/pretrained/chinese_L-12_H-768_A-12/"
BERT_VOCAB_PATH = os.path.join(BERT_PRETRAINED_PATH, 'vocab.txt')
BERT_CONFIG_PATH = os.path.join(BERT_PRETRAINED_PATH, "bert_config.json")


class TestBertModel(unittest.TestCase):

    def setUp(self):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB_PATH)
        self.config = BertConfig.from_pretrained(BERT_CONFIG_PATH)

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

    def test_bert_pretraining_model(self):
        # bert = BertForPretraining(self.config, training=True, trainable=True)
        bert = BertForPretraining.from_pretrained(pretrained_path=BERT_PRETRAINED_PATH,
                                                  training=True)

        bert.model.summary()

    def test_for_sequence_classification_model(self):
        # bert = BertForPretraining(elf.config, training=True, trainable=True)
        bert = BertForSequenceClassification.from_pretrained(pretrained_path=BERT_PRETRAINED_PATH,
                                                             training=True,
                                                             num_labels=2)

        bert.model.summary()
        input_ids = tf.constant([[7, 6, 0, 0, 0]], dtype=tf.int32)
        token_type_ids = tf.constant([[0, 0, 0, 0, 0]], dtype=tf.int32)
        ret = bert.model([input_ids, token_type_ids])  # build the network with dummy inputs
        print(ret)

    def ids_tensor(self, shape, vocab_size, rng=None, name=None, dtype=None):
        """Creates a random int32 tensor of the shape within the vocab size."""
        if rng is None:
            rng = random.Random()

        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(rng.randint(0, vocab_size - 1))

        output = tf.constant(values,
                             shape=shape,
                             dtype=dtype if dtype is not None else tf.int32)

        return output


if __name__ == '__main__':
    # python -m unittest test_modeling_bert.TestBertModel.test_encode
    unittest.main()
