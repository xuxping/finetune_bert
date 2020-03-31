# Quickstart


## Install
代码运行环境

Python3.5+
```
git clone https://github.com/mkavim/finetune_bert.git
cd finetune_bert
pip install .
```

## finetune 
```
from finetune import (BertConfig, BertTokenizer,BertForSequenceClassification)
from finetune.dataset import ChnSentiCorpDataset


tokenizer = MODELS[opts.model_name][1].from_pretrained(opts.pretrained_path)
dataset = ChnSentiCorpDataset(opts.data_dir, tokenizer, opts.max_seq_len)
X_train, y_train = dataset.get_train_datasets()
X_dev, y_dev = dataset.get_dev_datasets()
# build model
optimizer = tf.keras.optimizers.Adam(lr=opts.lr, epsilon=1e-08)

bert = BertForSequenceClassification.from_pretrained(
    pretrained_path=opts.pretrained_path,
    trainable=True,
    training=False,
    max_seq_len=256,
    num_labels=len(dataset.get_labels())
)

model = bert.model
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

model.fit(X_train, y_train,
            batch_size=opts.batch_size,
            epochs=opts.epochs,
            validation_data=(X_dev, y_dev),
            shuffle=True)

X_test, y_test = dataset.get_test_datasets()
score, acc = model.evaluate(X_test, y_test, batch_size=opts.batch_size)
print('test score:', score)
print('test accuracy:', acc)
```

## example    
1、 train   
目前只支持分类任务，其它任务将有计划的支持，见`examples/`   
```
python finetune_classifier.py --train  \
    --model_name bert \
    --task chnsenticorp \
    --pretrained_path=/xxx/chinese_L-12_H-768_A-12/ \
    --lr 3e-5 \
    --batch_size 32 \
    --epochs 3 \
    --save_dir ./checkpoint
```

2、test    
```
python finetune_classifier.py --test  \
    --task chnsenticorp \
    --pretrained_path=/xxx/chinese_L-12_H-768_A-12/vocab.txt \
    --batch_size 32 \
    --save_dir ./checkpoint/xxx.hdf5
```

3、tensorflow serving   
将keras bert 模型转化 tensorflow serving 的格式，即可上线   
```
python tools/keras_to_tf_serving.py \
    --model_path ./checkpoint/xxx.hdf5 \
    --export_model_dir ./tfserving_model/ \
    --model_version checkpoint_v1
```
