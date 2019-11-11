fine tune with keras-bert in classification task

### bert in keras
-------------
1、download bert pretrained model：[chinese_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)


2、finetune
```
python finetune_bert.py --train \
    --train_file ./examples/train.txt \
    --dev_file ./examples/dev.txt \
    --maxlen 128 \
    --lr 1e-5 \
    --batch_size 32 \
    --epochs=5 \
    --save_dir ./keras_bert01 \
    --vocab_file ~/.keras/datasets/chinese_L-12_H-768_A-12/vocab.txt
```

3、test
```
python finetune_bert.py --test \
    --test_file ./examples/dev.txt \
    --save_dir ./keras_bert01/05-0.9523.hdf5
```


4、Covert keras model to tensorflow `.pb` model
```
python keras_bert_to_tensorflow.py \
    --input_model ./keras_bert01/05-0.9523.hdf5 \  # input keras model
    --output_model bert.pb                         # output tensorflow model
```

5、 Covert keras model to tensorflow serving model
```
python keras_to_tf_serving.py \
    --model_path ./keras_bert01/05-0.9523.hdf5 \
    --export_model_dir ./tfserving_model/ \
    --model_version keras_bert_v1
```

### Reference:
--------------
1. [当Bert遇上Keras：这可能是Bert最简单的打开姿势](https://spaces.ac.cn/archives/6736)
2. [keras-bert](https://github.com/CyberZHG/keras-bert)
3. [keras_to_tensorflow](https://github.com/amir-abdi/keras_to_tensorflow)
