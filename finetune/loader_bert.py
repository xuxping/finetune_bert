# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)

    return _loader


def load_model_weights_from_checkpoint(model,
                                       config,
                                       checkpoint_file,
                                       training=False):
    """Load trained official model from checkpoint.

    :param model: Built keras model.
    :param config: Loaded configuration file.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    :param training: If training, the whole model will be returned.
                     Otherwise, the MLM and NSP parts will be ignored.
    """
    loader = checkpoint_loader(checkpoint_file)

    model.get_layer(name='Embedding-Token').set_weights([
        loader('bert/embeddings/word_embeddings'),
    ])
    model.get_layer(name='Embedding-Position').set_weights([
        loader('bert/embeddings/position_embeddings')[:config.max_position_embeddings, :],
    ])
    model.get_layer(name='Embedding-Segment').set_weights([
        loader('bert/embeddings/token_type_embeddings'),
    ])
    model.get_layer(name='Embedding-Norm').set_weights([
        loader('bert/embeddings/LayerNorm/gamma'),
        loader('bert/embeddings/LayerNorm/beta'),
    ])
    for i in range(config.num_hidden_layers):
        try:
            model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1))
        except ValueError as e:
            print("err Encoder-%d-MultiHeadSelfAttention" % (i + 1))
            raise e
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/attention/self/query/kernel' % i),
            loader('bert/encoder/layer_%d/attention/self/query/bias' % i),
            loader('bert/encoder/layer_%d/attention/self/key/kernel' % i),
            loader('bert/encoder/layer_%d/attention/self/key/bias' % i),
            loader('bert/encoder/layer_%d/attention/self/value/kernel' % i),
            loader('bert/encoder/layer_%d/attention/self/value/bias' % i),
            loader('bert/encoder/layer_%d/attention/output/dense/kernel' % i),
            loader('bert/encoder/layer_%d/attention/output/dense/bias' % i),
        ])
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
            loader('bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
        ])
        model.get_layer(name='Encoder-%d-FeedForward' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/intermediate/dense/kernel' % i),
            loader('bert/encoder/layer_%d/intermediate/dense/bias' % i),
            loader('bert/encoder/layer_%d/output/dense/kernel' % i),
            loader('bert/encoder/layer_%d/output/dense/bias' % i),
        ])
        model.get_layer(name='Encoder-%d-FeedForward-Norm' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/output/LayerNorm/gamma' % i),
            loader('bert/encoder/layer_%d/output/LayerNorm/beta' % i),
        ])
    model.get_layer(name='Pooler-Dense').set_weights([
        loader('bert/pooler/dense/kernel'),
        loader('bert/pooler/dense/bias'),
    ])

    if training:
        model.get_layer(name='MLM-Dense').set_weights([
            loader('cls/predictions/transform/dense/kernel'),
            loader('cls/predictions/transform/dense/bias'),
        ])
        model.get_layer(name='MLM-Norm').set_weights([
            loader('cls/predictions/transform/LayerNorm/gamma'),
            loader('cls/predictions/transform/LayerNorm/beta'),
        ])
        model.get_layer(name='MLM-Proba').set_weights([
            loader('cls/predictions/output_bias'),
        ])
        model.get_layer(name='NSP').set_weights([
            np.transpose(loader('cls/seq_relationship/output_weights')),
            loader('cls/seq_relationship/output_bias'),
        ])
