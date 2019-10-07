# -*- coding:utf-8 -*-
# Date: 2019/9/26 15:23
# Author: xuxiaoping
# Desc: Fine Tune with bert
import argparse
import codecs
import os

os.environ['TF_KERAS'] = '1'
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard

from keras_bert import Tokenizer
from tensorflow.python.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from keras_bert import get_custom_objects
from optimizers import AdamWarmup
from modeling_bert import BertForSequenceClassification
import collections
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def set_random():
    # seed
    import random
    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    fin = open(vocab_file)
    for num, line in enumerate(fin):
        token = line.strip()
        vocab[token] = num
    return vocab


class BertDataset(object):
    def __init__(self, data_path, opt=None):
        self.maxlen = opt.maxlen
        self.opt = opt
        token_dict = load_vocab(opts.vocab_file)
        self.tokenizer = Tokenizer(token_dict)

        self.token_ids, self.segment_ids, self.y = self._load_dataset(data_path)

    def _load_dataset(self, data_path):
        assert os.path.exists(data_path)
        token_ids, segment_ids = [], []
        y = []
        with codecs.open(data_path, 'r', encoding='utf-8') as fin:
            for lidx, line in enumerate(fin):
                try:
                    label, text = line.rstrip().split('\t')
                except Exception as e:
                    label = line.strip()
                    text = ''
                label = int(label)
                token_id, segment_id = self.tokenizer.encode(first=text, max_len=self.opt.maxlen)

                if lidx < 1:
                    print("text", text)
                    print("toke_id", token_id)
                    print("segment_id", segment_id)
                token_ids.append(token_id)
                segment_ids.append(segment_id)

                y.append([label])

        return token_ids, segment_ids, y

    def __len__(self):
        return len(self.y)

    def get_data(self):
        return [np.array(self.token_ids), np.array(self.segment_ids)], np.array(self.y)


def train(opts):
    print("load data...")
    X_train, y_train = BertDataset(opts.train_file, opts).get_data()
    X_dev, y_dev = BertDataset(opts.dev_file, opts).get_data()
    decay_steps = (len(y_train) // opts.batch_size) * opts.epochs

    optimizer = AdamWarmup(decay_steps=decay_steps,
                           warmup_steps=0,
                           lr=opts.lr,
                           weight_decay=0.0,
                           clipnorm=1.0)
    model = BertForSequenceClassification(opts)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())

    # callbacks: save model
    filepath = opts.save_dir + "/{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

    # callbacks: tensorboard
    tensorboard = TensorBoard(log_dir=opts.log_dir)

    model.fit(X_train, y_train,
              batch_size=opts.batch_size,
              epochs=opts.epochs,
              validation_data=(X_dev, y_dev),
              class_weight='auto',
              callbacks=[checkpoint, tensorboard])
    score, acc = model.evaluate(X_dev, y_dev, batch_size=opts.batch_size)
    print('dev score:', score)
    print('dev accuracy:', acc)


def get_predict(x, thresold=0.5):
    y_pred = []
    for pred in x:
        if pred[0] > thresold:
            y_pred.append(0)
        else:
            y_pred.append(1)
    return y_pred


def test(opts):
    X_test, y_test = BertDataset(opts.test_file, opts).get_data()

    # use get custiom_object to load model
    model = load_model(opts.save_dir, custom_objects=get_custom_objects())

    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred = get_predict(y_pred, opts.thresold)

    end_time = time.time()
    qps = float(len(y_test) + 1) / float(end_time - start_time + 1)
    print("qps: {}".format(qps))

    print(classification_report(y_test, y_pred, digits=4))
    print(confusion_matrix(y_test, y_pred))

    if opts.write_test_result:
        with codecs.open(opts.write_test_result, 'w', encoding='utf-8') as file_writer:
            for pred, label in zip(y_pred, y_test):
                file_writer.write('{}\t{}\n'.format(pred, label[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train', action='store_true', help='train mode')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--train_file', type=str, default=None,
                        help='train file')
    parser.add_argument('--dev_file', type=str, default=None,
                        help='dev file')
    parser.add_argument('--test_file', type=str, default=None,
                        help='test file')

    parser.add_argument('--bert_config_path', type=str,
                        default='~/.keras/datasets/chinese_L-12_H-768_A-12/bert_config.json',
                        help='bert config path')
    parser.add_argument('--bert_checkpoint_path', type=str,
                        default='~/.keras/datasets/chinese_L-12_H-768_A-12/bert_model.ckpt',
                        help='bert config path')
    parser.add_argument('--vocab_file', type=str,
                        default=None,
                        help='vocab file')

    parser.add_argument('--maxlen', type=int, default=128, help='max seq len')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='max seq len')
    parser.add_argument('--num_classes', type=int, default=2, help='num classes')
    parser.add_argument('--epochs', type=int, default=3, help='num classes')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='model save dir')
    parser.add_argument('--log_dir', type=str, default='./keras_logs', help='run log dir')
    parser.add_argument('--thresold', type=float, default=0.5, help='percision thresold')

    parser.add_argument('--write_test_result', type=str, default=None, help='file to write test result')

    opts = parser.parse_args()

    set_random()
    if opts.train:
        train(opts)

    if opts.test:
        test(opts)
