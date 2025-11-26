# src/tflite_infer.py
import argparse
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite  # or from tensorflow.lite import Interpreter on full TF
import os

def load_image(path, size=(128,128)):
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr

def infer(tflite_path, img_path, labels):
    # Load model
    interpreter = tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = load_image(img_path)
    inp = np.expand_dims(img, axis=0)

    # handle int8 models
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        inp = inp / scale + zero_point
        inp = inp.astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # dequantize if needed
    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        output_data = scale * (output_data.astype(np.float32) - zero_point)

    pred_idx = np.argmax(output_data)
    print("Predicted:", labels[pred_idx], "conf:", float(np.max(tf.nn.softmax(output_data))))

if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/recycle_model_quant.tflite")
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", default="labels.json")
    args = parser.parse_args()

    import json
    with open(args.labels) as f:
        labels = json.load(f)

    infer(args.model, args.image, labels)
