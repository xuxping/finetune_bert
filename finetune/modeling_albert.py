#! -*- coding: utf-8 -*-
"""
ALBert:
1、paper:https://arxiv.org/abs/1909.11942
2、code:https://github.com/google-research/ALBERT
3、pretrain file:https://storage.googleapis.com/albert_models/albert_base_zh.tar.gz
"""

import copy
import os

import tensorflow as tf

from finetune.activations import ACT2FN
from finetune.configuration_bert import ALBertConfig
from finetune.layers import get_initializer, shape_list, create_token_type_ids, create_position_ids, get_input_mask, \
    BiasAdd
from finetune.loader import load_albert_model_weights_from_checkpoint

try:
    LayerNormalization = tf.keras.layers.LayerNormalization
except AttributeError:
    from finetune.normalization import LayerNormalization

# pretrained file
ALBERT_CONFIG_NAME = 'config.json'
ALBERT_CHECKPOINT_NAME = 'model.ckpt'


class ALBertMultiHeadSelfAttention(tf.keras.layers.Layer):
    """Bert Multi-Head Self Attention.
    See https://github.com/huggingface/transformers/blob/master/transformers/modeling_tf_bert.py#L188-L257
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_probs_dropout_prob,
                 initializer_range,
                 **kwargs):
        super(ALBertMultiHeadSelfAttention, self).__init__(**kwargs)
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
        super(ALBertMultiHeadSelfAttention, self).build(input_shape)
        self.query = tf.keras.layers.Dense(self.all_head_size,
                                           kernel_initializer=get_initializer(self.initializer_range),
                                           name='query')
        self.key = tf.keras.layers.Dense(self.all_head_size,
                                         kernel_initializer=get_initializer(self.initializer_range),
                                         name='key')
        self.value = tf.keras.layers.Dense(self.all_head_size,
                                           kernel_initializer=get_initializer(self.initializer_range),
                                           name='value')

        self.dropout = tf.keras.layers.Dropout(rate=self.attention_probs_dropout_prob)

        self.linear = tf.keras.layers.Dense(self.all_head_size,
                                            kernel_initializer=get_initializer(self.initializer_range),
                                            name='linear')

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, **kwargs):
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
            # [batch_size, to_seq_length] -> [batch_size, 1, 1, to_seq_length]，在计算时会自动广播
            attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
            # 在softmax之前，考虑到attention_scores中可能含有负数，因此将padding乘上一个非常小的数，使得padding部分的影响趋于0
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

        # 加一层线性变换
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
        base_config = super(ALBertMultiHeadSelfAttention, self).get_config()
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
        # The activation is only applied to the "intermediate" hidden layer.
        self.intermediate = tf.keras.layers.Dense(self.intermediate_size,
                                                  kernel_initializer=get_initializer(self.initializer_range))

        self.intermediate_act_fn = ACT2FN[self.hidden_act]

        # Down-project back to `hidden_size` then add the residual.
        self.down_project = tf.keras.layers.Dense(self.hidden_size,
                                                  kernel_initializer=get_initializer(self.initializer_range))

    def call(self, inputs, **kwargs):
        hidden_states = self.intermediate(inputs)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.down_project(hidden_states)
        return hidden_states

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.hidden_size,)

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'hidden_act': self.hidden_act,
            'intermediate_size': self.intermediate_size,
            'initializer_range': self.initializer_range
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ALBertPretrained(object):
    def __init__(self, config, *inputs, **kwargs):
        self.model = None

    def build(self):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_path, trainable=True, training=False,
                        max_seq_len=None, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            config_file = os.path.join(pretrained_path, ALBERT_CONFIG_NAME)
            config = ALBertConfig.from_pretrained(config_file)
        bert = cls(config, trainable=trainable, training=training, max_seq_len=max_seq_len, **kwargs)
        bert.build()
        checkpoint_file = os.path.join(pretrained_path, ALBERT_CHECKPOINT_NAME)

        load_albert_model_weights_from_checkpoint(bert.model, config, checkpoint_file, training=training)
        return bert


class ALBertModel(ALBertPretrained):
    """构建跟Bert一样结构的Transformer-based模型
    """

    def __init__(self, config, trainable=True, training=False, max_seq_len=None, **kwargs):
        """
            trainable: 表示模型参数是否需要更新
            training: 是否是预训练模型
        """
        super(ALBertModel, self).__init__(config, trainable, training, max_seq_len, **kwargs)
        config = copy.deepcopy(config)
        if not isinstance(config, ALBertConfig):
            raise ValueError("config must be instance of BertConfig")

        self.trainable = trainable
        self.training = training
        if not training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.inner_group_num = config.inner_group_num
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.initializer_range = config.initializer_range or 0.02
        self.embedding_size = config.embedding_size
        self.hidden_act = config.hidden_act
        self.type_vocab_size = config.type_vocab_size
        self.layer_norm_eps = config.layer_norm_eps

        self.use_token_type = kwargs.pop('use_token_type', False)

        if max_seq_len and max_seq_len > config.max_position_embeddings:
            raise ValueError("max_seq_len < max_position_embeddings({})".format(config.max_position_embeddings))

        self.max_seq_len = max_seq_len
        self.build()

    def _embeddings(self, input_ids, token_type_ids=None, position_ids=None):
        self.share_token_embeddings = tf.keras.layers.Embedding(input_dim=self.vocab_size,
                                                                output_dim=self.embedding_size,
                                                                embeddings_initializer=get_initializer(
                                                                    self.initializer_range),
                                                                name='Embedding-Token')
        self.token_embeddings = self.share_token_embeddings(input_ids)
        if position_ids is None:
            position_ids = tf.keras.layers.Lambda(lambda x: create_position_ids(x),
                                                  name='Input-Position')(input_ids)

        if token_type_ids is None:
            token_type_ids = tf.keras.layers.Lambda(lambda x: create_token_type_ids(x))(input_ids)

        position_embeddings = tf.keras.layers.Embedding(input_dim=self.max_position_embeddings,
                                                        output_dim=self.embedding_size,
                                                        embeddings_initializer=get_initializer(self.initializer_range),
                                                        name='Embedding-Position')(position_ids)

        token_type_embeddings = tf.keras.layers.Embedding(input_dim=self.type_vocab_size,
                                                          output_dim=self.embedding_size,
                                                          embeddings_initializer=get_initializer(
                                                              self.initializer_range),
                                                          name='Embedding-Segment')(token_type_ids)
        embeddings = tf.keras.layers.Add(name='Embedding-Add')(
            [self.token_embeddings, position_embeddings, token_type_embeddings])

        embeddings = LayerNormalization(epsilon=self.layer_norm_eps, name='Embedding-Norm')(embeddings)
        embeddings = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob, name='Embedding-Dropout')(embeddings)

        return embeddings

    def _trainable(self, _layer):
        if isinstance(self.trainable, (list, tuple, set)):
            for prefix in self.trainable:
                if _layer.name.startswith(prefix):
                    return True
            return False
        return self.trainable

    def get_pooled_output(self):
        # pooled_output字段存放句子粒度的特征，可用于文本分类等任务
        return self.pooler_output

    def get_sequence_output(self):
        # sequence_output字段存放字粒度的特征，可用于序列标注等任务
        return self.all_layer_outputs[-1]

    def get_all_layer_outputs(self):
        # 所有层的输出
        return self.all_layer_outputs

    def get_token_embeddings(self):
        return self.token_embeddings

    def embedding_similarity(self, inputs):
        batch_size = shape_list(inputs)[0]
        length = shape_list(inputs)[1]

        def reshape1(inputs):
            return tf.reshape(inputs, [-1, self.hidden_size])

        def reshape2(inputs):
            return tf.reshape(inputs, [batch_size, length, self.vocab_size])

        def matmul(x):
            return tf.matmul(x, self.share_token_embeddings.embeddings, transpose_b=True)

        inputs = tf.keras.layers.Lambda(lambda x: reshape1(x))(inputs)
        logits = tf.keras.layers.Lambda(lambda x: matmul(x))(inputs)

        return tf.keras.layers.Lambda(lambda x: reshape2(x))(logits)

    def build(self):
        """ALBert模型构建函数"""
        input_ids = tf.keras.layers.Input(shape=(self.max_seq_len,), name='Input-Token')
        model_inputs = [input_ids]
        if self.use_token_type:
            token_type_ids = tf.keras.layers.Input(shape=(self.max_seq_len,), name='Input-Segment')
            model_inputs.append(token_type_ids)
        else:
            token_type_ids = tf.keras.layers.Lambda(lambda x: create_token_type_ids(x),
                                                    name='Input-Segment')(input_ids)

        # attention_mask和bert中的input_mask一致
        attention_mask = tf.keras.layers.Lambda(lambda x: get_input_mask(x), name="Attention-Mask")(input_ids)
        embeddings = self._embeddings(input_ids, token_type_ids)

        # embedding 因式分解
        if self.embedding_size != self.hidden_size:
            embeddings = tf.keras.layers.Dense(self.hidden_size,
                                               kernel_initializer=get_initializer(self.initializer_range),
                                               name="Factor-Dense")(embeddings)

        # 主要Transformer Encoder部分
        self.all_layer_outputs = []

        prev_output = embeddings
        layers = None
        for i in range(self.num_hidden_layers):
            attention_name = 'Encoder-MultiHeadSelfAttention'
            feed_forward_name = 'Encoder-FeedForward'
            encoder_output, layers = self.transformer_block(
                inputs=prev_output,
                attention_mask=attention_mask,
                attention_name=attention_name,
                feed_forward_name=feed_forward_name,
                layers=layers  # 层复用，tensorflow代码中采用的variable_scope来复用变量，keras直接复用就行
            )
            self.all_layer_outputs.append(encoder_output)
            prev_output = encoder_output

        # pooler，取[CLS]的输出做一次线性变换，用于句子或者句对的分类
        sequence_output = self.all_layer_outputs[-1]
        first_token_tensor = tf.keras.layers.Lambda(lambda x: x[:, 0], name='Pooler')(sequence_output)
        self.pooler_output = tf.keras.layers.Dense(self.hidden_size,
                                                   activation='tanh',
                                                   kernel_initializer=get_initializer(self.initializer_range),
                                                   name="Pooler-Dense")(first_token_tensor)

        # sequence_output, pooler_output
        outputs = [sequence_output, self.pooler_output]

        self.model = tf.keras.Model(model_inputs, outputs)
        for layer in self.model.layers:
            layer.trainable = self._trainable(layer)

    def transformer_block(self, inputs, attention_mask=None, attention_name='attention',
                          feed_forward_name='feed-forward', layers=None):
        """构建单个Transformer Block"""
        x = inputs
        if layers is None:
            layers = [
                ALBertMultiHeadSelfAttention(hidden_size=self.hidden_size,
                                             num_attention_heads=self.num_attention_heads,
                                             initializer_range=self.initializer_range,
                                             attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                                             name=attention_name),
                tf.keras.layers.Dropout(rate=self.hidden_dropout_prob,
                                        name='%s-Dropout' % attention_name),
                tf.keras.layers.Add(name='%s-Add' % attention_name),
                LayerNormalization(epsilon=self.layer_norm_eps, name='%s-Norm' % attention_name),
                FeedForward(intermediate_size=self.intermediate_size,
                            hidden_size=self.hidden_size,
                            hidden_act=self.hidden_act,
                            initializer_range=self.initializer_range,
                            name=feed_forward_name),
                tf.keras.layers.Dropout(rate=self.hidden_dropout_prob,
                                        name='%s-Dropout' % feed_forward_name),
                tf.keras.layers.Add(name='%s-Add' % feed_forward_name),
                LayerNormalization(epsilon=self.layer_norm_eps, name='%s-Norm' % feed_forward_name),
            ]
        # Self Attention
        xi = x
        x = layers[0]([x, attention_mask])
        # dropout
        x = layers[1](x)

        # Add & Norm
        x = layers[2]([xi, x])
        x = layers[3](x)

        # Feed Forward
        xi = x
        x = layers[4](x)
        x = layers[5](x)
        # Add & Norm
        x = layers[6]([xi, x])
        x = layers[7](x)
        return x, layers


class ALBertForPretraining(ALBertPretrained):
    """用于对ALBert进行预训练"""

    def __init__(self, config, trainable=True, training=True, max_seq_len=None, **kwargs):
        super(ALBertForPretraining, self).__init__(config, trainable, training, max_seq_len, **kwargs)
        self.bert = ALBertModel(config, trainable=trainable, training=training, max_seq_len=max_seq_len, **kwargs)
        self.input_embeddings = self.bert.get_token_embeddings()

        # NSP
        self.seq_relationship = tf.keras.layers.Dense(2,
                                                      kernel_initializer=get_initializer(config.initializer_range),
                                                      name='NSP')
        # MLM
        self.mlm_dense = tf.keras.layers.Dense(config.hidden_size,
                                               kernel_initializer=get_initializer(config.initializer_range),
                                               name='MLM-Dense')
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = LayerNormalization(epsilon=config.layer_norm_eps, name='MLM-Norm')
        self.bais_add = BiasAdd(initializer_range=config.initializer_range, name='MLM-Proba')

    def build(self):
        sequence_out, pooler_out = self.bert.model.output

        # MLM
        hidden_states = self.mlm_dense(sequence_out)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        # [batch_size, length, self.vocab_size]
        hidden_states = self.bert.embedding_similarity(hidden_states)

        prediction_scores = self.bais_add(hidden_states)

        # SOP: sentence order predict
        seq_relationship_scores = self.seq_relationship(pooler_out)
        output = [seq_relationship_scores, prediction_scores]
        self.model = tf.keras.Model(self.bert.model.input, output)


class ALBertForSequenceClassification(ALBertPretrained):
    # 句子或者句对分类(use_token_type)
    def __init__(self, config, trainable=True, training=False, max_seq_len=None, **kwargs):
        super(ALBertForSequenceClassification, self).__init__(config, trainable, training, max_seq_len, **kwargs)
        self.bert = ALBertModel(config, trainable=trainable, training=training, max_seq_len=max_seq_len, **kwargs)
        num_labels = int(kwargs.pop('num_labels', 2))
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob, name='classifier-drop')
        self.classifier = tf.keras.layers.Dense(units=num_labels,
                                                activation='softmax',
                                                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                                kernel_initializer=get_initializer(config.initializer_range),
                                                name="classifier")

    def build(self, **kwargs):
        pooler_out = self.bert.get_pooled_output()
        output = self.dropout(pooler_out)
        output = self.classifier(output)

        self.model = tf.keras.Model(self.bert.model.input, output)


class ALBertFoTokenClassification(ALBertPretrained):
    # For Sequence Tag Task

    def __init__(self, config, trainable=True, training=False, max_seq_len=None, **kwargs):
        super(ALBertFoTokenClassification, self).__init__(config, trainable, training, max_seq_len, **kwargs)
        self.bert = ALBertModel(config, trainable=trainable, training=training, max_seq_len=max_seq_len, **kwargs)
        num_labels = int(kwargs.pop('num_labels', 2))
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob, name='classifier-drop')

        self.classifier = tf.keras.layers.Dense(units=num_labels,
                                                activation='softmax',
                                                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                                kernel_initializer=get_initializer(config.initializer_range),
                                                name="classifier")

    def build(self, **kwargs):
        sequence_output = self.bert.get_sequence_output()
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        self.model = tf.keras.Model(self.bert.model.input, logits)


class ALBertForQuestionAnswering(ALBertPretrained):
    # For Question Answering Task

    def __init__(self, config, trainable=True, training=False, max_seq_len=None, **kwargs):
        super(ALBertForQuestionAnswering, self).__init__(config, trainable, training, max_seq_len, **kwargs)
        self.bert = ALBertModel(config, trainable=trainable, training=training, max_seq_len=max_seq_len, **kwargs)
        num_labels = int(kwargs.pop('num_labels', 2))
        self.qa_outputs = tf.keras.layers.Dense(units=num_labels,
                                                activation='softmax',
                                                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                                kernel_initializer=get_initializer(config.initializer_range),
                                                name="qa_outputs")

    def build(self, **kwargs):
        sequence_output = self.bert.get_sequence_output()
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.keras.layers.Lambda(lambda x: tf.split(x, 2, axis=-1))(logits)
        start_logits = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(start_logits)
        end_logits = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(end_logits)

        self.model = tf.keras.Model(self.bert.model.input, [start_logits, end_logits])


custom_objects = {
    'ALBertMultiHeadSelfAttention': ALBertMultiHeadSelfAttention,
    'LayerNormalization': LayerNormalization,
    'FeedForward': FeedForward,
}

tf.keras.utils.get_custom_objects().update(custom_objects)
