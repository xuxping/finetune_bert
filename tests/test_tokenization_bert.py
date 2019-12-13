import unittest

from finetune.tokenization_bert import BertTokenizer


class MyTestCase(unittest.TestCase):
    def test_encode(self):
        tokenizer = BertTokenizer.from_pretrained('../bert_config/vocab.txt')
        text_a = "今天天气很不错噢"
        token_ids, segment_ids, input_mask = tokenizer.encode(text_a=text_a, max_seq_length=128)
        print(token_ids)
        print(segment_ids)
        print(input_mask)

    def test_pair_encode(self):
        tokenizer = BertTokenizer.from_pretrained('../bert_config/vocab.txt')
        text_a = "今天天气很不错噢"
        text_b = "明天天气也是很不错的噢"
        token_ids, segment_ids, input_mask = tokenizer.encode(text_a=text_a, text_b=text_b, max_seq_length=128)
        print(token_ids)
        print(segment_ids)
        print(input_mask)


if __name__ == '__main__':
    unittest.main()
