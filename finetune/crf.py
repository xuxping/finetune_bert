# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K


def viterbi_decode(score, trans_matrix):
    """viterbi decode

      Args:
        score: A [seq_len, num_tags] matrix of unary potentials.
        trans_matrix: A [num_tags, num_tags] matrix of binary potentials.

      Returns:
        viterbi: path
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + trans_matrix
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    return np.asarray(viterbi)


class CRF(tf.keras.layers.Layer):
    """Condition Random Field.
    本质是一个loss
    """

    def __init__(self,
                 **kwargs):
        super(CRF, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CRF, self).build(input_shape)
        num_labels = input_shape[0][-1].value
        print(num_labels)
        self.trans = self.add_weight(
            name='transition',
            shape=(num_labels, num_labels),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs, **kwargs):
        # hidden_states:(B, T, N)
        # mask:(B, T)
        hidden_states, mask = inputs
        # mask掉pad的部分
        mask = mask[:, :, tf.newaxis]  # (B, T) -> (B, T, N)
        addr = (1. - tf.cast(mask, tf.float32)) * -1e12
        return hidden_states + addr

    def sparse_loss(self, y_true, y_pred):
        """
        Args:
            y_true: (B, T), one-hot
            y_pred: (B, T, N)

        Returns: crf loss
        """
        y_true = tf.reshape(y_true, tf.shape(y_pred)[:-1])
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, tf.shape(self.trans)[0])  # (-1, seq_len, num_labels)
        return self.dense_loss(y_true, y_pred)

    def get_E(self, y_true, y_pred):
        """E = emition score + trans score
        本质是计算真实路径的得分
        """
        # 输出的标签与真实标签的得分 s(y_i, X, i)
        emition_score = tf.reduce_sum(y_true * y_pred, 2)  # (B, T)
        # # 从T-1时刻，输出的标签，并转移到T时刻 t(y_{i}, y_{i-1}, X, i)
        trans_score = tf.reduce_sum(tf.tensordot(y_true[:, :-1, :], self.trans, 1) * y_true[:, 1:, :], 2)  # (B, T-1)
        return tf.reduce_sum(emition_score, -1) + tf.reduce_sum(trans_score, -1)

    def log_norm_step(self, inputs, states):
        """参考：https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L629"""
        inputs, mask = inputs[:, :-1], inputs[:, -1:]
        states = tf.expand_dims(states[0], 2)  # (B, N, 1)
        trans = tf.expand_dims(self.trans, 0)  # (1, N, N)
        outputs = tf.reduce_logsumexp(
            states + trans, 1
        )  # (batch_size, output_dim)
        outputs = outputs + inputs
        return outputs, [outputs]

    def get_logZ(self, y_pred, mask):
        """计算所有路径的得分, 采用递归进行计算"""
        previous = [y_pred[:, 0]]  # 初始状态
        input_length = tf.shape(y_pred)[1]

        y_pred = tf.concat([y_pred, mask], axis=2)
        log_norm, _, _ = K.rnn(
            self.log_norm_step,
            y_pred[:, 1:],
            previous,
            input_length=input_length
        )

        return tf.reduce_logsumexp(log_norm, 1)  # (B, 1)

    def dense_loss(self, y_true, y_pred):
        """
        loss = logZ - E, 所有路径的得分 - 真实路径的得分，越小越好
        """
        # 这里的y_pred已经在call()中被Mask过
        mask = tf.reduce_all(tf.greater(y_pred, -1e6),
                             axis=2,
                             keep_dims=True)  # [B, T, N]
        mask = tf.cast(mask, tf.float32)
        # mask = [1,1,1,1,1,0,0,0,0,...]
        y_true = y_true * mask
        y_pred = y_pred * mask

        E = self.get_E(y_true, y_pred)

        logZ = self.get_logZ(y_pred, mask)

        return logZ - E

    def compute_mask(self, inputs, mask=None):
        return None

    @staticmethod
    def sparse_accuracy(y_true, y_pred):
        """近似代替viterbi的得分
        Args:
            y_true:  (B, T)
            y_pred:  (B, T, N)

        Returns: accuracy
        """
        mask = tf.cast(tf.reduce_all(tf.greater(y_pred, -1e6), 2), tf.float32)  # [B, T]
        y_pred = tf.cast(tf.argmax(y_pred, 2), tf.int32)

        y_true = tf.reshape(y_true, tf.shape(y_pred))
        y_true = tf.cast(y_true, tf.int32)
        isequal = tf.cast(tf.equal(y_true, y_pred), tf.float32)

        # 排除mask的影响
        return tf.reduce_sum(isequal * mask) / tf.reduce_sum(mask)


custom_objects = {
    'CRF': CRF,
}

tf.keras.utils.get_custom_objects().update(custom_objects)
