# -*- coding:utf-8 -*-
# Date: 2019/12/26
# Author: xuxiaoping01
# Desc:
import tensorflow as tf


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def create_token_type_ids(input_ids):
    input_shape = shape_list(input_ids)
    token_type_ids = tf.fill(input_shape, 0)
    return token_type_ids


def create_position_ids(input_ids):
    input_shape = shape_list(input_ids)
    seq_length = input_shape[1]
    position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
    return position_ids


def get_input_mask(input_ids):
    return tf.greater(input_ids, 0)


def get_initializer(initializer_range=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


class BiasAdd(tf.keras.layers.Layer):
    def __init__(self,
                 initializer_range,
                 **kwargs):
        super(BiasAdd, self).__init__(**kwargs)
        self.initializer_range = initializer_range

    def get_config(self):
        config = {
            'initializer_range': self.initializer_range
        }
        base_config = super(BiasAdd, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.bias = self.add_weight(
            shape=(input_shape[-1].value,),
            initializer=get_initializer(self.initializer_range),
            name='bias',
        )
        super(BiasAdd, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        return tf.nn.bias_add(inputs, self.bias)


custom_objects = {
    'BiasAdd': BiasAdd,
}

tf.keras.utils.get_custom_objects().update(custom_objects)
