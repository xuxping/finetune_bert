为什么想要自己实现实现一个bert:   
1. 熟悉，从论文到参考别人的代码，到自己走一遍，才能深入了解  
2. 可控，基于Bert改进的模型太多太杂，需要自己整理一个框架处理，供后面的研究    
3. 自定义，增加一些其他需求，如蒸馏等等    

采用tensorflow的keras实现，目前仅支持tensorflow1.13.1+，暂不考虑支持tensorflow2.0+

### 未来3个月计划（2019/12-2020/02）  
1、基于bert打造一个通用的NLP算法底层框架（pretrain+finetune）   
2、增加模型蒸馏能力  
3、增加更多的使用样例(分类，NER，相似度，QA等)   

### 更新
* 20191229(计划): 支持albert模型
* 20191222(计划): 支持一种蒸馏模型
* 20191214: 调整modeling_bert的结构，统一from_pretrained   
* 20191208: 与官方bert进行对比，修复load_vocab中词id错位的问题，对chnsenticorp进行finetune结果达到了94%左右，与官方一致 
* 20191207: 结构微调，实现的过程中逐步加深对bert的理解，目前对chnsenticorp数据集finetune的结果只有90%左右，存在潜在问题未解决
* 20191206: 初步实现加载官方bert模型，不支持自己训练Bert，增加chnsenticorp文本分类示例    

### 测试结果

|dataset | dev acc | test acc | batch size | learing rate | train epoch |
| ---- | ---- | ---- | ---- | ---- | ---- |
|chnsenticorp|94.00%|94.08%|32|3e-5|4|

**测试结果未精调，仅供效果实现参考**

### 如何使用  
1、下载Bert中文预训练模型：[chinese_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

2、finetune
```
finetune_classifier.py --train  \
    --pretrained_path=/xxx/chinese_L-12_H-768_A-12/ \
    --lr 3e-5 \
    --batch_size 32 \
    --epochs 3 \
    --save_dir ./keras_bert/
```

3、test
```
finetune_classifier.py --test  \
    --pretrained_path=/xxx/chinese_L-12_H-768_A-12/vocab.txt \
    --batch_size 32 \
    --save_dir ./keras_bert/xxx.hdf5
```


4、 将keras bert 模型转化 tensorflow serving 的格式
```
python keras_to_tf_serving.py \
    --model_path ./keras_bert01/05-0.9523.hdf5 \
    --export_model_dir ./tfserving_model/ \
    --model_version keras_bert_v1
```

### 参考资料:
1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
2. [google-bert](https://github.com/google-research/bert)
3. [当Bert遇上Keras：这可能是Bert最简单的打开姿势](https://spaces.ac.cn/archives/6736)
4. [bert4keras](https://github.com/bojone/bert4keras)
5. [keras-bert](https://github.com/CyberZHG/keras-bert)
6. [keras_to_tensorflow](https://github.com/amir-abdi/keras_to_tensorflow)
7. https://github.com/huggingface/transformers