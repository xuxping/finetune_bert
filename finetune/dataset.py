# -*- coding:utf-8 -*-

import codecs
import csv
import numpy as np
import os


class Dataset(object):
    def __init__(self, data_dir, tokenizer, max_seq_len=512):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def preproccess(self, lines):
        token_ids, segment_ids, labels = [], [], []
        for lidx, line in enumerate(lines):
            try:
                label = int(line[0])
            except:
                print("[err]: {}".format(line))
                raise
            token_id, segment_id, input_mask = self.tokenizer.encode(text_a=line[1], max_seq_length=self.max_seq_len)
            token_ids.append(token_id)
            segment_ids.append(segment_id)
            labels.append(label)
            if lidx < 1:
                print("text", line[1])
                print("label", line[0])
                print("token_id", token_id)
                print("segment_id", segment_id)
                print("input_mask", input_mask)
        return self.get_data(token_ids, segment_ids, labels)

    @staticmethod
    def _read_dataset(input_file, quotechar=None):
        with codecs.open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            next(reader)  # skip header
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def get_train_datasets(self):
        raise NotImplementedError

    def get_dev_datasets(self):
        raise NotImplementedError

    def get_test_datasets(self):
        raise NotImplementedError

    @staticmethod
    def get_data(token_ids, segment_ids, labels):
        return [np.array(token_ids, dtype=np.int32),
                np.array(segment_ids, dtype=np.int32)], np.array(labels, dtype=np.int32)

    def get_labels(self):
        return []


class ChnSentiCorpDataset(Dataset):

    def get_train_datasets(self):
        lines = self._read_dataset(os.path.join(self.data_dir, 'train.tsv'))
        return self.preproccess(lines)

    def get_dev_datasets(self):
        lines = self._read_dataset(os.path.join(self.data_dir, 'dev.tsv'))
        return self.preproccess(lines)

    def get_test_datasets(self):
        lines = self._read_dataset(os.path.join(self.data_dir, 'test.tsv'))
        return self.preproccess(lines)

    def get_labels(self):
        return [0, 1]
