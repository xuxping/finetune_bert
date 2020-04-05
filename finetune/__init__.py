# -*- coding:utf-8 -*-

__version__ = '0.0.1'

# config
from .configuration_bert import BertConfig, DistillBertConfig, ALBertConfig
from .modeling_albert import ALBertModel, ALBertForSequenceClassification, ALBertForPretraining

# model
from .modeling_bert import BertModel, BertForSequenceClassification, BertForPretraining
from .modeling_distilbert import DistillBertModel, DistillBertForSequenceClassification, DistillBertForPretraining

# tokenizer
from .tokenization_bert import BertTokenizer
from .tokenization_distillbert import DistillBertTokenizer
from .tokenization_albert import ALBertTokenizer


MODEL_ZOOS = {
    "bert": (BertConfig,
             BertTokenizer,
             BertModel,
             BertForPretraining,
             BertForSequenceClassification),
    "albert": (ALBertConfig,
               ALBertTokenizer,
               ALBertModel,
               ALBertForPretraining,
               ALBertForSequenceClassification),
    'distillbert': (
        DistillBertConfig,
        DistillBertTokenizer,
        DistillBertModel,
        DistillBertForPretraining,
        DistillBertForSequenceClassification)
}
