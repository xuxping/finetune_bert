# -*- coding:utf-8 -*-

import sys
import unittest
import os

sys.path.append("../")
from finetune.dataset import Sst2Dataset
from finetune.tokenization_distillbert import DistillBertTokenizer

GLUE_DATA_DIR = '../datasets/glue_data'


class TestDataset(unittest.TestCase):

    def test_sst2_dataset(self):
        tokenizer = DistillBertTokenizer('../configs/distillbert/vocab.txt',
                                         do_lower_case=True)
        datasets = Sst2Dataset(os.path.join(GLUE_DATA_DIR, 'sst-2'), tokenizer)

        datasets.get_train_datasets()


if __name__ == '__main__':
    unittest.main()
