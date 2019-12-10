#! -*- coding: utf-8 -*-
# 实现参考:
# https://github.com/google-research/bert
# https://github.com/huggingface/transformers
# https://github.com/bojone/bert4keras


import numpy as np
import tensorflow as tf
import copy
from finetune.loader import load_model_weights_from_checkpoint
from finetune.configuration_bert import BertConfig
from finetune.normalization import LayerNormalization


def get_initializer(initializer_range=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def gelu(x):
    """ Gaussian Error Linear Unit.
    Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf


def gelu_new(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(x):
    return x * tf.sigmoid(x)


ACT2FN = {"gelu": tf.keras.layers.Activation(gelu),
          "relu": tf.keras.activations.relu,
          "swish": tf.keras.layers.Activation(swish),
          "gelu_new": tf.keras.layers.Activation(gelu_new)}


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """Bert Multi-Head Self Attention."""

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_probs_dropout_prob,
                 initializer_range,
                 **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def build(self, input_shape):
        super(MultiHeadSelfAttention, self).build(input_shape)
        self.query = tf.keras.layers.Dense(self.all_head_size,
                                           kernel_initializer=get_initializer(self.initializer_range),
                                           name='query')
        self.key = tf.keras.layers.Dense(self.all_head_size,
                                         kernel_initializer=get_initializer(self.initializer_range),
                                         name='key')
        self.value = tf.keras.layers.Dense(self.all_head_size,
                                           kernel_initializer=get_initializer(self.initializer_range),
                                           name='value')

        self.dropout = tf.keras.layers.Dropout(self.attention_probs_dropout_prob)

        self.linear = tf.keras.layers.Dense(self.all_head_size,
                                            kernel_initializer=get_initializer(self.initializer_range),
                                            name='linear')

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        hidden_states, attention_mask = inputs

        batch_size = tf.shape(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(query_layer, key_layer,
                                     transpose_b=True)  # (batch size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(key_layer)[-1], tf.float32)  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
            attention_scores = attention_scores + adder

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer,
                                   (batch_size, -1, self.all_head_size))  # (batch_size, seq_len_q, all_head_size)

        output = self.linear(context_layer)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.all_head_size)

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'attention_probs_dropout_prob': self.attention_probs_dropout_prob,
            'initializer_range': self.initializer_range
        }
        base_config = super(MultiHeadSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, intermediate_size,
                 hidden_size,
                 hidden_act,
                 initializer_range, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.hidden_act = hidden_act

    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)

        self.dense_1 = tf.keras.layers.Dense(self.intermediate_size,
                                             kernel_initializer=get_initializer(self.initializer_range))

        self.intermediate_act_fn = ACT2FN[self.hidden_act]

        self.dense_2 = tf.keras.layers.Dense(self.hidden_size,
                                             kernel_initializer=get_initializer(self.initializer_range))

    def call(self, inputs):
        hidden_states = self.dense_1(inputs)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'hidden_act': self.hidden_act,
            'intermediate_size': self.intermediate_size,
            'initializer_range': self.initializer_range
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def create_token_type_ids(tokens_input):
    input_shape = shape_list(tokens_input)
    token_type_ids = tf.fill(input_shape, 0)
    return token_type_ids


def create_position_ids(tokens_input):
    input_shape = shape_list(tokens_input)
    seq_length = input_shape[1]
    position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
    return position_ids


def get_input_mask(tokens_input):
    return tf.greater(tokens_input, 0)


class BertModel(object):
    """构建跟Bert一样结构的Transformer-based模型
    """

    def __init__(self, config, trainable=True, training=False):
        config = copy.deepcopy(config)
        if not isinstance(config, BertConfig):
            raise ValueError("config must be instance of BertConfig")

        self.trainable = trainable

        if not training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.initializer_range = config.initializer_range or 0.02
        self.embedding_size = config.hidden_size
        self.hidden_act = config.hidden_act
        self.type_vocab_size = config.type_vocab_size
        self.config = config

    def _embeddings(self, tokens_input, token_type_input=None, position_input=None):
        # Embedding部分
        tokens_embeddings = tf.keras.layers.Embedding(input_dim=self.vocab_size,
                                                      output_dim=self.embedding_size,
                                                      embeddings_initializer=get_initializer(self.initializer_range),
                                                      name='Embedding-Token')(tokens_input)
        if position_input is None:
            position_input = tf.keras.layers.Lambda(lambda x: create_position_ids(x))(tokens_input)

        if token_type_input is None:
            token_type_input = tf.keras.layers.Lambda(lambda x: create_token_type_ids(x))(tokens_input)

        position_embeddings = tf.keras.layers.Embedding(input_dim=self.max_position_embeddings,
                                                        output_dim=self.embedding_size,
                                                        embeddings_initializer=get_initializer(self.initializer_range),
                                                        name='Embedding-Position')(position_input)

        token_type_embeddings = tf.keras.layers.Embedding(input_dim=self.type_vocab_size,
                                                          output_dim=self.embedding_size,
                                                          embeddings_initializer=get_initializer(
                                                              self.initializer_range),
                                                          name='Embedding-Segment')(token_type_input)
        embeddings = tf.keras.layers.Add(name='Embedding-Add')(
            [tokens_embeddings, position_embeddings, token_type_embeddings])

        embeddings = LayerNormalization(name='Embedding-Norm')(embeddings)
        self.embeddings = tf.keras.layers.Dropout(self.hidden_dropout_prob, name='Embedding-Dropout')(embeddings)
        return self.embeddings

    def get_embeddings_out(self):
        return self.embeddings

    def _trainable(self, _layer):
        if isinstance(self.trainable, (list, tuple, set)):
            for prefix in self.trainable:
                if _layer.name.startswith(prefix):
                    return True
            return False
        return self.trainable

    def build(self):
        """Bert模型构建函数"""
        # 设置输入
        tokens_input = tf.keras.layers.Input(shape=(None,), name='Input-Token')
        token_type_input = tf.keras.layers.Input(shape=(None,), name='Input-Segment')

        layer_out = self._embeddings(tokens_input, token_type_input)

        # 主要Transformer Encoder部分
        all_layer_outputs = []
        attention_mask = tf.keras.layers.Lambda(lambda x: get_input_mask(x))(tokens_input)

        for i in range(self.num_hidden_layers):
            attention_name = 'Encoder-%d-MultiHeadSelfAttention' % (i + 1)
            feed_forward_name = 'Encoder-%d-FeedForward' % (i + 1)
            layer_out = self.transformer_block(
                inputs=layer_out,
                attention_mask=attention_mask,
                attention_name=attention_name,
                feed_forward_name=feed_forward_name)
            layer_out = self.post_processing(i, layer_out)
            all_layer_outputs.append(layer_out)

        # 最后一层的输出
        outputs = [layer_out]

        # Pooler
        first_token_embeddings = tf.keras.layers.Lambda(lambda x: x[:, 0], name='Pooler')(layer_out)
        self.pooler_output = tf.keras.layers.Dense(self.hidden_size,
                                                   activation='tanh',
                                                   kernel_initializer=get_initializer(self.initializer_range),
                                                   name="Pooler-Dense")(first_token_embeddings)
        outputs.append(self.pooler_output)

        self.model = tf.keras.Model([tokens_input, token_type_input], outputs)
        for layer in self.model.layers:
            layer.trainable = self._trainable(layer)

    def get_pooled_output(self):
        # [CLS] for classification
        return self.pooler_output

    def transformer_block(self, inputs, attention_mask=None, attention_name='attention',
                          feed_forward_name='feed-forward'):
        """构建单个Transformer Block
        """
        x = inputs
        layers = [
            MultiHeadSelfAttention(hidden_size=self.hidden_size,
                                   num_attention_heads=self.num_attention_heads,
                                   initializer_range=self.initializer_range,
                                   attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                                   name=attention_name),
            tf.keras.layers.Dropout(rate=self.hidden_dropout_prob,
                                    name='%s-Dropout' % attention_name),
            tf.keras.layers.Add(name='%s-Add' % attention_name),
            LayerNormalization(name='%s-Norm' % attention_name),
            FeedForward(intermediate_size=self.intermediate_size,
                        hidden_size=self.hidden_size,
                        hidden_act=self.hidden_act,
                        initializer_range=self.initializer_range,
                        name=feed_forward_name),
            tf.keras.layers.Dropout(rate=self.hidden_dropout_prob,
                                    name='%s-Dropout' % feed_forward_name),
            tf.keras.layers.Add(name='%s-Add' % feed_forward_name),
            LayerNormalization(name='%s-Norm' % feed_forward_name),
        ]
        # Self Attention
        xi = x
        x = layers[0]([x, attention_mask])
        if self.hidden_dropout_prob > 0:
            x = layers[1](x)

        # 残差连接
        x = layers[2]([xi, x])
        x = layers[3](x)

        # Feed Forward
        xi = x
        x = layers[4](x)
        if self.hidden_dropout_prob > 0:
            x = layers[5](x)
        x = layers[6]([xi, x])
        x = layers[7](x)
        return x

    def post_processing(self, layer_id, inputs):
        """自定义每一个block的后处理操作
        """
        return inputs

    @classmethod
    def from_pretrained(cls, bert_config_path, bert_checkpoint_path, trainable=True, training=False):
        config = BertConfig.from_pretrained(bert_config_path)
        bert = cls(config, trainable=trainable, training=training)
        bert.build()
        load_model_weights_from_checkpoint(bert.model, config, bert_checkpoint_path)
        return bert


class BertForSequenceClassification(object):

    @staticmethod
    def build(config):
        bert = BertModel.from_pretrained(bert_config_path=config.bert_config_path,
                                         bert_checkpoint_path=config.bert_checkpoint_path,
                                         trainable=True)

        layers_out, pooler_out = bert.model.output
        output = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)(pooler_out)
        output = tf.keras.layers.Dense(units=config.num_labels,
                                       activation='softmax',
                                       kernel_initializer=get_initializer(config.initializer_range))(output)

        return tf.keras.Model(bert.model.input, output)


custom_objects = {
    'MultiHeadSelfAttention': MultiHeadSelfAttention,
    'LayerNormalization': LayerNormalization,
    'FeedForward': FeedForward,
}

tf.keras.utils.get_custom_objects().update(custom_objects)
