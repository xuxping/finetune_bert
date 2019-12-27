# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import h5py
import os


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)

    return _loader


def load_bert_model_weights_from_checkpoint(model,
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


def load_distillbert_model_weights_from_checkpoint(model,
                                                   config,
                                                   checkpoint_file,
                                                   training=False):
    # loader weight form huggingface:
    # https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-tf_model.h5
    assert os.path.exists(checkpoint_file)
    f = h5py.File(checkpoint_file, 'r')

    def loader(name, prefix='distilbert/tf_distil_bert_for_masked_lm'):
        weight_name = '{}/{}:0'.format(prefix, name)
        print('load {}'.format(weight_name))
        return np.asarray(f[weight_name])

    model.get_layer(name='Embedding-Token').set_weights([
        loader('distilbert/embeddings/word_embeddings/weight')
    ])
    model.get_layer(name='Embedding-Position').set_weights([
        loader('distilbert/embeddings/position_embeddings/embeddings')[:config.max_position_embeddings, :]
    ])
    model.get_layer(name='Embedding-Norm').set_weights([
        loader('distilbert/embeddings/LayerNorm/gamma'),
        loader('distilbert/embeddings/LayerNorm/beta'),
    ])
    for i in range(config.num_hidden_layers):
        try:
            model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1))
        except ValueError as e:
            print("err Encoder-%d-MultiHeadSelfAttention" % (i + 1))
            raise e
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1)).set_weights([
            loader('distilbert/transformer/layer_._%d/attention/q_lin/kernel' % i),
            loader('distilbert/transformer/layer_._%d/attention/q_lin/bias' % i),
            loader('distilbert/transformer/layer_._%d/attention/k_lin/kernel' % i),
            loader('distilbert/transformer/layer_._%d/attention/k_lin/bias' % i),
            loader('distilbert/transformer/layer_._%d/attention/v_lin/kernel' % i),
            loader('distilbert/transformer/layer_._%d/attention/v_lin/bias' % i),
            loader('distilbert/transformer/layer_._%d/attention/out_lin/kernel' % i),
            loader('distilbert/transformer/layer_._%d/attention/out_lin/bias' % i),
        ])
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
            loader('distilbert/transformer/layer_._%d/sa_layer_norm/gamma' % i),
            loader('distilbert/transformer/layer_._%d/sa_layer_norm/beta' % i),
        ])
        model.get_layer(name='Encoder-%d-FeedForward' % (i + 1)).set_weights([
            loader('distilbert/transformer/layer_._%d/ffn/lin1/kernel' % i),
            loader('distilbert/transformer/layer_._%d/ffn/lin1/bias' % i),
            loader('distilbert/transformer/layer_._%d/ffn/lin2/kernel' % i),
            loader('distilbert/transformer/layer_._%d/ffn/lin2/bias' % i),
        ])
        model.get_layer(name='Encoder-%d-FeedForward-Norm' % (i + 1)).set_weights([
            loader('distilbert/transformer/layer_._%d/output_layer_norm/gamma' % i),
            loader('distilbert/transformer/layer_._%d/output_layer_norm/beta' % i),
        ])

    if training:
        prefix = 'vocab_transform/tf_distil_bert_for_masked_lm'
        model.get_layer(name='MLM-Dense').set_weights([
            loader('vocab_transform/kernel', prefix=prefix),
            loader('vocab_transform/bias', prefix=prefix),
        ])
        prefix = 'vocab_layer_norm/tf_distil_bert_for_masked_lm'
        model.get_layer(name='MLM-Norm').set_weights([
            loader('vocab_layer_norm/gamma', prefix=prefix),
            loader('vocab_layer_norm/beta', prefix=prefix),
        ])
        prefix = 'vocab_projector/tf_distil_bert_for_masked_lm'
        model.get_layer(name='MLM-Proba').set_weights([
            loader('vocab_projector/bias', prefix=prefix),
        ])
