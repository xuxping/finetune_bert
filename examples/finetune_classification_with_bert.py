# -*- coding:utf-8 -*-
import argparse
import codecs
import os
import random
import sys

os.environ['TF_KERAS'] = '1'
sys.path.append('../')
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
# from modeling_bert import BertForSequenceClassification
from finetune.modeling_bert import BertForSequenceClassification
from tensorflow.python.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from finetune.optimizers import AdamWarmup
from finetune.tokenization_bert import BertTokenizer
from finetune.dataset import Dataset
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_random():
    # seed
    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)


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


def train(opts):
    tokenizer = BertTokenizer(opts.vocab_file)
    dataset = ChnSentiCorpDataset(opts.data_dir, tokenizer, opts.maxlen)

    X_train, y_train = dataset.get_train_datasets()
    X_dev, y_dev = dataset.get_dev_datasets()
    decay_steps = (len(y_train) // opts.batch_size) * opts.epochs
    print("decay_steps==", decay_steps)
    optimizer = AdamWarmup(decay_steps=decay_steps,
                           warmup_steps=0,
                           lr=opts.lr,
                           weight_decay=0.0,
                           clipnorm=1.0)

    model = BertForSequenceClassification().build(opts)
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
    tokenizer = BertTokenizer(opts.vocab_file)
    dataset = ChnSentiCorpDataset(opts.data_dir, tokenizer, opts.maxlen)
    X_test, y_test = dataset.get_test_datasets()

    # use get custiom_object to load model
    model = load_model(opts.save_dir)

    start_time = time.time()

    y_pred = model.predict(X_test, batch_size=opts.batch_size)
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
    parser.add_argument('--test_threshold', action='store_true', help='test mode')
    parser.add_argument('--data_dir', type=str, default='../datasets/chnsenticorp')

    parser.add_argument('--bert_config_path', type=str,
                        default=None,
                        help='bert config path')
    parser.add_argument('--bert_checkpoint_path', type=str,
                        default=None,
                        help='bert config path')
    parser.add_argument('--vocab_file', type=str,
                        default=None,
                        help='vocab file')

    parser.add_argument('--maxlen', type=int, default=128, help='max seq len')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, help='max seq len')
    parser.add_argument('--num_labels', type=int, default=2, help='num labels')
    parser.add_argument('--initializer_range', type=float, default=0.02, help='initializer range')
    parser.add_argument('--epochs', type=int, default=3, help='train epochs')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='model save dir')
    parser.add_argument('--log_dir', type=str, default='./logs', help='run log dir')
    parser.add_argument('--thresold', type=float, default=0.5, help='percision thresold')

    parser.add_argument('--write_test_result', type=str, default=None, help='file to write test result')

    opts = parser.parse_args()

    set_random()
    if opts.train:
        train(opts)

    if opts.test:
        test(opts)
