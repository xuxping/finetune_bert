使用keras-bert 在分类任务上进行finetune

### bert in keras
-------------
1、下载Bert中文预训练模型：[chinese_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

2、finetune
```
!python finetune_bert.py --train \
    --train_file ./examples/train.txt \
    --dev_file ./examples/dev.txt \
    --maxlen 128 \
    --lr 5e-5 \
    --batch_size 32 \
    --epochs=5 \
    --save_dir ./keras_bert01 \
    --vocab_file ~/.keras/datasets/chinese_L-12_H-768_A-12/vocab.txt \
    --bert_config_path ~/.keras/datasets/chinese_L-12_H-768_A-12/bert_config.json \
    --bert_checkpoint_path ~/.keras/datasets/chinese_L-12_H-768_A-12/bert_model.ckpt
```

3、test
```
!python finetune_bert.py --test \
    --test_file ./examples/dev.txt \
    --save_dir ./keras_bert01/03-0.9493.hdf5 \
    --vocab_file ~/.keras/datasets/chinese_L-12_H-768_A-12/vocab.txt
```


4、将keras bert 模型转化 tensorflow 的pb格式， 参考[keras_to_tensorflow](kerashttps://github.com/amir-abdi/keras_to_tensorflow)进行修改后使用
```
python keras_bert_to_tensorflow.py \
    --input_model ./keras_bert01/03-0.9493.hdf5 \  # 输入keras模型路径
    --output_model bert.pb                         # 输出pb模型路径
```


### 使用建议：
-------
- 采用tensorflow 1.12.0版本，并将TF_KERAS设置为'1'
- 使用keras的tensorboard callback时，将write_grads=False
- load_model时，需要设置custom_objects为keras-bert的get_custom_objects(), 不然会报错

### 参考资料:
-----------
1. [当Bert遇上Keras：这可能是Bert最简单的打开姿势](https://spaces.ac.cn/archives/6736)
2. [keras-bert](https://github.com/CyberZHG/keras-bert)
3. [keras_to_tensorflow](https://github.com/amir-abdi/keras_to_tensorflow)
