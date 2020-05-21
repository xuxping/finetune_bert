为什么想要自己实现实现一个finetune代码框架:
1. 熟悉，从论文到参考别人的代码，到自己走一遍，才能深入了解
2. 可控，基于Bert改进的模型太多太杂，需要自己整理一个框架处理，供后面的研究
3. 自定义，增加一些其他需求，如知识蒸馏和模型压缩等等
4. 简单化，有些包支持模型过多，代码越来越复杂，太难看懂

采用tensorflow的keras实现，目前仅支持tensorflow1.13.1+。本文实现很多参考了[transformers](https://github.com/huggingface/transformers)和[bert4keras](https://github.com/bojone/bert4keras)，在此表示感谢。

### 已支持功能
* 中文BERT、ALBert
* 英文版distillbert

### 计划支持
* ELECTRA
* TinyBERT

**由于时间限制，更新不会太快，需要读懂论文，然后实现和测试等，基本保持一个月支持一个模型**


### 更新 
2020:
* **2020/04/09**: 增加NER任务的支持，在chinadaily数据集上test_f1达到0.95301,可参考`examples/finetune_ner`
* **2020/04/05**: 增加albert在chnsenticorp的测试
* **2020/03/31**: 支持albert,采用谷歌发布的中文版本,并在lcmqc上进行测试,增加项目文档
* **2020/02/08**: 提供可安装的Python包   
* **2020/01/03**: 增加在[LCQMC](http://icrc.hitsz.edu.cn/info/1037/1146.htm)数据集上的句对分类示例  

2019:  
* **2019/12/28:** 初步实现distillbert，并成功加载官方模型权重，对一些变量名进行更新，修复bert实现中一个错误，增加sst-2任务  
* **2019/12/14:** 调整modeling_bert的结构，统一from_pretrained   
* **2019/12/08:** 与官方bert进行对比，修复load_vocab中词id错位的问题，对chnsenticorp进行finetune结果达到了94%左右，与官方一致 
* **2019/12/07:** 结构微调，实现的过程中逐步加深对bert的理解，目前对chnsenticorp数据集finetune的结果只有90%左右，存在潜在问题未解决
* **2019/12/06:** 初步实现加载官方bert模型，不支持自己训练Bert，增加chnsenticorp文本分类示例    

### 测试结果

|model |dataset | dev acc | test acc | batch size | learing rate | train epoch |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |  
|bert|chnsenticorp|94.00|94.08|32|3e-5|4|  
|bert|LCQMC|88.74|86.11|32|3e-5|4|
|albert|chnsenticorp|93.17|93.58|32|3e-5|6|
|albert|LCQMC|87.25|87.06|32|3e-5|4|
|distillbert|sst-2|90.71|90.38|32|4e-5|4|

**测试结果未精调，仅供测试模型是否正确，能否达到预期的效果**

### 数据集
|dataset | description |
| ---- | ---- |  
|chnsenticorp|中文情感正负面分析语料，包含三类：旅馆、书籍、商品评论|  
|LCQMC|中文句对相似度任务|
|chinadaily|中文实体标注数据|
|sst-2|英文正负面情绪分类任务|


### 安装
需要提前安装依赖包，参考requirement.txt

```
git clone https://github.com/mkavim/finetune_bert.git
cd finetune_bert
pip install .
```

### 如何使用

1、下载Bert中文预训练模型：[chinese_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

2、finetune
支持classification和ner任务，参考`examples`

```
python finetune_classifier.py --train  \
    --model_name bert \
    --task chnsenticorp \
    --pretrained_path=./chinese_L-12_H-768_A-12/ \
    --lr 3e-5 \
    --batch_size 32 \
    --epochs 3 \
    --save_dir ./checkpoint
```

3、test
```
python finetune_classifier.py --test  \
    --task chnsenticorp \
    --pretrained_path=./chinese_L-12_H-768_A-12/vocab.txt \
    --batch_size 32 \
    --save_dir ./checkpoint/xxx.hdf5
```

4、 将keras bert 模型转化 tensorflow serving 的格式，即可上线
```
python tools/keras_to_tf_serving.py \
    --model_path ./checkpoint/xxx.hdf5 \
    --export_model_dir ./tfserving_model/ \
    --model_version keras_bert_xxx
```

### 一些问题
1. 为什么只支持tf.keras?   
    一是keras未来的趋势是与tensorflow整合，因此直接使用tf.keras实现。低版本keras不支持sublayers，具体可参考[issues](https://github.com/keras-team/keras/issues/11653)。
    另外tf2.0版本太高，实际环境中还不敢使用

2. 是否可以提交代码和issues?
   个人精力和能力有限，欢迎提交代码和issues

3. 模型支持计划？
    * 后面倾向于支持小的预训练模型，实用性更高，尽量保持每个月支持一个模型或者一个新的功能
    * 主要做中文NLP，优先支持中文预训练模型   
    * web可视化建模开发  


### 文章产出
待定

### 参考资料  
1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)  
2. https://github.com/google-research/bert  
3. https://github.com/bojone/bert4keras   
4. https://github.com/CyberZHG/keras-bert   
5. https://github.com/huggingface/transformers   
6. https://github.com/amir-abdi/keras_to_tensorflow   
7. [当Bert遇上Keras：这可能是Bert最简单的打开姿势](https://spaces.ac.cn/archives/6736)
