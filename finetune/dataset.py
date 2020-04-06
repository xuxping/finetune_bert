"""Dataset"""

import codecs
import csv
import os

import numpy as np


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


class LcqmcDataset(Dataset):
    """LCQMC Dataset."""

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

    def preproccess(self, lines):
        token_ids, segment_ids, labels = [], [], []
        for lidx, line in enumerate(lines):
            if len(line) != 3:
                print("err {}".format(line))
                continue
            token_id, segment_id, input_mask = self.tokenizer.encode(text_a=line[0], text_b=line[1],
                                                                     max_seq_length=self.max_seq_len)
            label = int(line[2])
            token_ids.append(token_id)
            segment_ids.append(segment_id)
            labels.append(label)
            if lidx < 1:
                print("text", line)
                print("token_id", token_id)
                print("segment_id", segment_id)
                print("input_mask", input_mask)
        return self.get_data(token_ids, segment_ids, labels)


class Sst2Dataset(Dataset):
    """GLUE SST-2 Dataset."""

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


class ChinaDailyDataset(Dataset):
    """china people daily ner corpus"""

    def __init__(self, data_dir, tokenizer, max_seq_len=512):
        super(ChinaDailyDataset, self).__init__(data_dir, tokenizer, max_seq_len)
        # tag map
        tags_dict = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']

        self.tags_map = {v: k for k, v in enumerate(tags_dict)}

    @staticmethod
    def _read_dataset(input_file, quotechar=None):
        sentences = []
        sentence = []
        with codecs.open(input_file, 'r', 'utf8') as freader:
            for lidx, line in enumerate(freader):
                line = line.rstrip()
                cols = line.split()
                if not cols:
                    if len(sentence) > 0:
                        sentences.append(sentence)
                    sentence = []
                else:
                    if len(cols) == 2:
                        sentence.append(cols)
                    else:
                        print('err:' + line)
        # [[[word, tag],[word,tag],[word, tag],...]]
        return sentences

    def preproccess(self, lines):
        token_ids, segment_ids, labels = [], [], []
        for lidx, line in enumerate(lines):
            sentence = ''.join([item[0] for item in line])
            # [CLS]....[SEP]
            label_seq = [0] + [self.tags_map[item[1]] for item in line] + [0]
            # 用-1来表示label pad，后面会用于计算sentence的真实长度
            label_seq = label_seq + [-1] * (self.max_seq_len - len(label_seq))
            label_seq = label_seq[:self.max_seq_len]

            token_id, segment_id, input_mask = self.tokenizer.encode(text_a=sentence, max_seq_length=self.max_seq_len)

            token_ids.append(token_id)
            segment_ids.append(segment_id)
            labels.append(label_seq)
            if lidx < 1:
                print("text", sentence)
                print("label", label_seq)
                print("token_id", token_id)
                print("segment_id", segment_id)
                print("input_mask", input_mask)
        return self.get_data(token_ids, segment_ids, labels)

    def get_train_datasets(self):
        lines = self._read_dataset(os.path.join(self.data_dir, 'example.train'))
        return self.preproccess(lines)

    def get_dev_datasets(self):
        lines = self._read_dataset(os.path.join(self.data_dir, 'example.dev'))
        return self.preproccess(lines)

    def get_test_datasets(self):
        lines = self._read_dataset(os.path.join(self.data_dir, 'example.test'))
        return self.preproccess(lines)

    def get_labels(self):
        return [v for _, v in self.tags_map.items()]
