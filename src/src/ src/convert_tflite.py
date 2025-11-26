# src/convert_tflite.py
import tensorflow as tf
import numpy as np

# Load saved model
keras_model = tf.keras.models.load_model("models/recycle_model.h5")

# Basic float32 conversion
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()
open("models/recycle_model.tflite", "wb").write(tflite_model)

# Post-training integer quantization for edge devices (recommended)
# Needs a representative dataset generator
def representative_data_gen():
    for _ in range(100):
        # yield a batch dimension (1, H, W, C) of float32 input
        data = np.random.rand(1, 128, 128, 3).astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set input/output to int8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
open("models/recycle_model_quant.tflite", "wb").write(tflite_quant_model)
