# -*- coding:utf-8 -*-
import argparse
import codecs
import os
import random
import sys
import time
from datetime import datetime

sys.path.append('../')

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from finetune import MODEL_ZOOS
from finetune.dataset import ChnSentiCorpDataset, Sst2Dataset, LcqmcDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_random():
    # seed
    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)


TASK_NAMES = {
    'chnsenticorp': ChnSentiCorpDataset,
    'lcqmc': LcqmcDataset,
    'sst-2': Sst2Dataset,
}



def train(opts):
    tokenizer = MODEL_ZOOS[opts.model_name][1].from_pretrained(opts.pretrained_path)

    # get dataset
    dataset = TASK_NAMES[opts.task_name](opts.data_dir, tokenizer, opts.max_seq_len)
    X_train, y_train = dataset.get_train_datasets()
    X_dev, y_dev = dataset.get_dev_datasets()
    if not opts.use_token_type or opts.model_name == 'distillbert':
        X_train = X_train[0]
        X_dev = X_dev[0]
    # build model
    optimizer = tf.keras.optimizers.Adam(lr=opts.lr, epsilon=1e-08)

    # GPU should support mixed precision and tensorflow version >= 1.14+
    if opts.use_fp16:
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    bert = MODEL_ZOOS[opts.model_name][4].from_pretrained(
        pretrained_path=opts.pretrained_path,
        trainable=True,
        training=False,
        max_seq_len=opts.max_seq_len,
        num_labels=len(dataset.get_labels())
    )

    model = bert.model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # callbacks: save model
    filepath = os.path.join(opts.save_dir, "{epoch:02d}-{val_acc:.4f}.hdf5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False,
                                                    mode='max')

    # callbacks: tensorboard
    tensorboard_dir = os.path.join(opts.log_dir, datetime.now().strftime("%Y%m%d-%H%M"))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

    model.fit(X_train, y_train,
              batch_size=opts.batch_size,
              epochs=opts.epochs,
              validation_data=(X_dev, y_dev),
              shuffle=True,
              callbacks=[checkpoint, tensorboard])

    X_test, y_test = dataset.get_test_datasets()
    score, acc = model.evaluate(X_test, y_test, batch_size=opts.batch_size)
    print('test score:', score)
    print('test accuracy:', acc)


def test(opts):
    tokenizer = MODEL_ZOOS[opts.model_name][1].from_pretrained(opts.pretrained_path)
    dataset = TASK_NAMES[opts.task_name](opts.data_dir, tokenizer, opts.max_seq_len)
    X_test, y_test = dataset.get_test_datasets()
    if not opts.use_token_type or opts.model_name == 'distillbert':
        X_test = X_test[0]
    # use get custiom_object to load model
    model = tf.keras.models.load_model(opts.save_dir)

    start_time = time.time()
    y_pred = model.predict(X_test, batch_size=opts.batch_size)
    y_pred = np.argmax(y_pred, axis=1)

    end_time = time.time()
    qps = float(len(y_test) + 1) / float(end_time - start_time + 1)
    print("qps: {}".format(qps))

    print(classification_report(y_test, y_pred, digits=4))
    print(confusion_matrix(y_test, y_pred))

    if opts.write_test_result:
        with codecs.open(opts.write_test_result, 'w', encoding='utf-8') as file_writer:
            for pred, label, token_id in zip(y_pred, y_test, X_test[0]):
                tokens = tokenizer.convert_ids_to_tokens(token_id)
                file_writer.write('{}\t{}\t{}\n'.format(pred, label, ''.join(tokens)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', action='store_true', help='train mode')
    parser.add_argument('--test', action='store_true', help='test mode')

    parser.add_argument('--use_fp16', action='store_true', help='use float16 mixed precision')

    parser.add_argument('--model_name', type=str, default='bert', choices=MODEL_ZOOS.keys())
    parser.add_argument('--task_name', type=str, default='chnsenticorp', choices=TASK_NAMES.keys())
    parser.add_argument('--data_dir', type=str, default='../datasets/chnsenticorp')
    parser.add_argument('--pretrained_path', type=str, default=None, help='bert pretrained path')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max seq len')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--use_token_type', type=int, default=0, help='use_token_type')
    parser.add_argument('--epochs', type=int, default=4, help='train epochs')
    parser.add_argument('--save_dir', type=str, default=None, help='model save dir')
    parser.add_argument('--log_dir', type=str, default='./logs', help='tensorboard log dir')

    parser.add_argument('--write_test_result', type=str, default=None, help='file to write test result')

    opts = parser.parse_args()
    assert opts.task_name in TASK_NAMES
    print(opts.__dict__)
    set_random()
    if opts.train:
        train(opts)

    if opts.test:
        test(opts)
