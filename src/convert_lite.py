import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #If the line below doesn't work, uncomment this line (make sure to comment the line below); it should help.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
'''
Numbers for "os.environ['TF_CPP_MIN_LOG_LEVEL']": 
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
import tensorflow as tf
from termcolor import colored

def tensorflow_to_lite():
    saved_model_dir = "./tf_models"
    lite_model_dir = "./lite_models"

    if not os.path.exists(saved_model_dir):
        print("\n\nNo model found!")
    else:
        # Convert the model
        print("\n\nModel found! Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
        tflite_model = converter.convert()

        # Save the model
        if not os.path.exists(lite_model_dir):
                    os.makedirs(lite_model_dir)
        with open('./lite_models/model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print("\n\nConverted to TFLite!")

tensorflow_to_lite()