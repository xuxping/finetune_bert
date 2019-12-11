# -*- coding:utf-8 -*-
import argparse
import codecs
import os
import random
import sys

sys.path.append('../')
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from finetune.modeling_bert import BertForSequenceClassification
from tensorflow.python.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from finetune.optimizers import AdamWarmup
from finetune.tokenization_bert import BertTokenizer
from finetune.dataset import ChnSentiCorpDataset
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_random():
    # seed
    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)


def train(opts):
    tokenizer = BertTokenizer.from_pretrained(opts.pretrained_path)
    dataset = ChnSentiCorpDataset(opts.data_dir, tokenizer, opts.max_seq_len)

    X_train, y_train = dataset.get_train_datasets()
    X_dev, y_dev = dataset.get_dev_datasets()
    decay_steps = (len(y_train) // opts.batch_size) * opts.epochs
    print("decay_steps==", decay_steps)
    optimizer = AdamWarmup(decay_steps=decay_steps,
                           warmup_steps=0,
                           lr=opts.lr,
                           weight_decay=0.01,
                           clipnorm=1.0)

    model = BertForSequenceClassification().build(opts)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())

    # callbacks: save model
    filepath = os.path.join(opts.save_dir, "{epoch:02d}-{val_acc:.4f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

    # callbacks: tensorboard
    tensorboard = TensorBoard(log_dir=opts.log_dir)

    model.fit(X_train, y_train,
              batch_size=opts.batch_size,
              epochs=opts.epochs,
              validation_data=(X_dev, y_dev),
              callbacks=[checkpoint, tensorboard])

    X_test, y_test = dataset.get_test_datasets()
    score, acc = model.evaluate(X_test, y_test, batch_size=opts.batch_size)
    print('test score:', score)
    print('test accuracy:', acc)


def get_predict(x, thresold=0.5):
    y_pred = []
    for pred in x:
        if pred[0] > thresold:
            y_pred.append(0)
        else:
            y_pred.append(1)
    return y_pred


def test(opts):
    tokenizer = BertTokenizer.from_pretrained(opts.pretrained_path)
    dataset = ChnSentiCorpDataset(opts.data_dir, tokenizer, opts.max_seq_len)
    X_test, y_test = dataset.get_test_datasets()

    # use get custiom_object to load model
    model = load_model(opts.save_dir)

    start_time = time.time()

    y_pred = model.predict(X_test, batch_size=opts.batch_size)
    y_pred = np.argmax(y_pred)

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

    parser.add_argument('--pretrained_path', type=str, default=None, help='bert pretrained path')

    parser.add_argument('--max_seq_len', type=int, default=256, help='max seq len')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=24, help='max seq len')
    parser.add_argument('--num_labels', type=int, default=2, help='num labels')
    parser.add_argument('--initializer_range', type=float, default=0.02, help='initializer range')
    parser.add_argument('--epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='model save dir')
    parser.add_argument('--log_dir', type=str, default='./logs', help='run log dir')

    parser.add_argument('--write_test_result', type=str, default=None, help='file to write test result')

    opts = parser.parse_args()

    set_random()
    if opts.train:
        train(opts)

    if opts.test:
        test(opts)
