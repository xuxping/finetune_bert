# -*- coding:utf-8 -*-
# Desc: Covert keras model to tf serving model

import os
from argparse import ArgumentParser

os.environ['TF_KERAS'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from keras_bert import get_custom_objects


def export_model(model_path, export_model_dir, model_version):
    """Export keras model to tf serving model for production environments
    see:
        https://github.com/jefferyUstc/MnistOnKeras/blob/master/export_model.py
    """
    model = load_model(model_path, custom_objects=get_custom_objects())

    with tf.get_default_graph().as_default():
        # prediction_signature
        tensor_info_input = tf.saved_model.utils.build_tensor_info(model.input)
        tensor_info_output = tf.saved_model.utils.build_tensor_info(model.output)
        print(model.output.shape, '**', tensor_info_output)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'text': tensor_info_input},  # Tensorflow.TensorInfo
                outputs={'result': tensor_info_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        print('step1 => prediction_signature created successfully')
        # set-up a builder
        export_path_base = export_model_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(model_version)))
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        builder.add_meta_graph_and_variables(
            # tags:SERVING,TRAINING,EVAL,GPU,TPU
            sess=K.get_session(),
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={'prediction_signature': prediction_signature, },
        )
        print('step2 => Export path(%s) ready to export trained model' % export_path, '\n starting to export model...')
        builder.save(as_text=True)
        print('Done exporting!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True)
    parser.add_argument('--export_model_dir', type=str, default=None, required=True)
    parser.add_argument('--model_version', type=str, default=0.1, required=True)

    opts = parser.parse_args()

    export_model(
        opts.model_path,
        opts.export_model_dir,
        opts.model_version
    )
