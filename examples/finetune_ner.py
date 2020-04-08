"""Finetune Ner Task

train:
-------------------------------------------
python finetune_ner.py --train \
    --model_name albert \
    --pretrained_path ../configs/albert/ \
    --task_name chinadaily \
    --max_seq_len 128

test:
-------------------------------------------
python finetune_ner.py --test \
    --model_name albert \
    --pretrained_path ../configs/albert/ \
    --task_name chinadaily \
    --max_seq_len 128   \
    --save_dir checkpoint/best_model.weights


1、use sparse_accuracy and sparse_loss, epochs=4
dev:  f1: 0.95301, precision: 0.96605, recall: 0.94032
test:  f1: 0.95137, precision: 0.95710, recall: 0.94571

2、use crf_viterbi_accuracy and crf_sparse_loss, epochs=6
def f1: 0.95494, precision: 0.95874, recall: 0.95117
test:  f1: 0.94872, precision: 0.94802, recall: 0.94943
"""

import argparse
import os
import random
import sys

sys.path.append('../')

import numpy as np
import tensorflow as tf

from finetune import MODEL_ZOOS
from finetune.dataset import ChinaDailyDataset
from finetune.crf import CRF, viterbi_decode, crf_sparse_loss, crf_viterbi_accuracy
from finetune.layers import get_input_mask
from tensorflow.python.keras import backend as K

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_random():
    # seed
    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)


TASK_NAMES = {
    'chinadaily': ChinaDailyDataset,
}


class Evaluate(tf.keras.callbacks.Callback):

    def __init__(self, validation_data, model, crf, opts):
        super(Evaluate, self).__init__()
        self.validation_data = validation_data
        self.model = model
        self.best_val_f1 = 0
        self.opts = opts
        self.crf = crf

    def on_epoch_end(self, epoch, logs=None):
        X_dev, y_dev = self.validation_data
        preds = self.model.predict(X_dev)  # [B, T, N]
        X, Y, Z = 0, 0, 0
        trans_matrix = K.eval(self.crf.trans)

        for i, score in enumerate(preds):
            rel_len = len([j for j in y_dev[i] if j != -1])  # real sequence length
            pred_path = viterbi_decode(score[:rel_len], trans_matrix)

            rel_path = y_dev[i][1:rel_len - 1]  # exclude [CLS], [SEP]

            pred_path = pred_path[1:-1]

            assert len(pred_path) == len(rel_path)

            pred = set([tag for tag in pred_path if tag > 0])  # exclude 'O'
            rel = set([tag for tag in rel_path if tag > 0])

            X += len(pred & rel)
            Y += len(pred)
            Z += len(rel)

        precision = X / Y
        recall = X / Z
        f1 = 2 * X / (Y + Z)

        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights(os.path.join(opts.save_dir, 'best_model.weights'))
            # self.model.save(os.path.join(self.opts.save_dir, 'best_model.h5'))

        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


def get_ner_model(opts, num_labels):
    # Transformer + Dense + CRF
    bert = MODEL_ZOOS[opts.model_name][2].from_pretrained(
        pretrained_path=opts.pretrained_path,
        trainable=True,
        training=False,
        max_seq_len=opts.max_seq_len,
        num_labels=num_labels
    )

    sequence_output = bert.get_sequence_output()
    input_mask = tf.keras.layers.Lambda(get_input_mask,
                                        name="Input-Mask")(bert.model.inputs[0])
    project_dense = tf.keras.layers.Dense(num_labels, name='Project-Dense')
    scores = project_dense(sequence_output)
    crf = CRF(name='crf_loss')
    crf_out = crf([scores, input_mask])

    model = tf.keras.Model(bert.model.input, crf_out)
    return model, crf


def train(opts):
    tokenizer = MODEL_ZOOS[opts.model_name][1].from_pretrained(opts.pretrained_path)

    dataset = TASK_NAMES[opts.task_name](opts.data_dir, tokenizer, opts.max_seq_len)
    X_train, y_train = dataset.get_train_datasets()
    X_dev, y_dev = dataset.get_dev_datasets()

    optimizer = tf.keras.optimizers.Adam(lr=opts.lr, epsilon=1e-08)

    num_labels = len(dataset.get_labels())
    model, crf = get_ner_model(opts, num_labels)
    model.summary()

    model.compile(
        optimizer=optimizer,
        loss=crf_sparse_loss,
        metrics=[crf_viterbi_accuracy]
    )

    evaluate = Evaluate((X_dev, y_dev), model, crf, opts)

    model.fit(X_train, y_train,
              batch_size=opts.batch_size,
              epochs=opts.epochs,
              validation_data=(X_dev, y_dev),
              shuffle=True,
              callbacks=[evaluate])


def test(opts):
    tokenizer = MODEL_ZOOS[opts.model_name][1].from_pretrained(opts.pretrained_path)
    dataset = TASK_NAMES[opts.task_name](opts.data_dir, tokenizer, opts.max_seq_len)
    X_test, y_test = dataset.get_test_datasets()

    model, crf = get_ner_model(opts, len(dataset.get_labels()))
    model.load_weights(opts.save_dir)

    trans_matrix = K.eval(crf.trans)
    preds = model.predict(X_test)
    X, Y, Z = 0, 0, 0
    for i, score in enumerate(preds):
        rel_len = len([j for j in y_test[i] if j != -1])  # real sequence length
        pred_path = viterbi_decode(score[:rel_len], trans_matrix)

        rel_path = y_test[i][1:rel_len - 1]  # exclude [CLS], [SEP]

        pred_path = pred_path[1:-1]

        assert len(pred_path) == len(rel_path)

        pred = set([tag for tag in pred_path if tag > 0])  # exclude 'O'
        rel = set([tag for tag in rel_path if tag > 0])

        X += len(pred & rel)
        Y += len(pred)
        Z += len(rel)

    precision = X / Y
    recall = X / Z
    f1 = 2 * X / (Y + Z)
    print(
        'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
        (f1, precision, recall)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', action='store_true', help='train mode')
    parser.add_argument('--test', action='store_true', help='test mode')

    parser.add_argument('--use_fp16', action='store_true', help='use float16 mixed precision')

    parser.add_argument('--model_name', type=str, default='albert', choices=MODEL_ZOOS.keys())
    parser.add_argument('--task_name', type=str, default='chinadaily', choices=TASK_NAMES.keys())
    parser.add_argument('--data_dir', type=str, default='../datasets/chinadaily')
    parser.add_argument('--pretrained_path', type=str, default='../configs/albert', help='bert pretrained path')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max seq len')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--use_token_type', type=int, default=0, help='use_token_type')
    parser.add_argument('--epochs', type=int, default=4, help='train epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoint', help='model save dir')
    parser.add_argument('--log_dir', type=str, default='./logs', help='tensorboard log dir')

    opts = parser.parse_args()
    assert opts.task_name in TASK_NAMES
    print(opts.__dict__)
    set_random()
    if opts.train:
        train(opts)

    if opts.test:
        test(opts)
