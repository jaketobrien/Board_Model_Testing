##### MODEL CONVERSION (onnx -> tflite) #####

import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model_path = r"C:\Users\Realt\Documents\Model_Testing\Saved_models\Isolation_Forest_P-3_50.onnx"

try:
    onnx_model = onnx.load(onnx_model_path)
except Exception as e:
    print("Error loading ONNX model:", e)
    exit(1)

# Prepare the ONNX model for TensorFlow conversion
try:
    tf_rep = prepare(onnx_model)
except Exception as e:
    print("Error preparing ONNX model:", e)
    exit(1)

if tf_rep is None:
    print("Error: TensorFlow representation is None")
    exit(1)

# Convert the model to TensorFlow format
tf_model_path = r"C:\Users\Realt\Documents\Model_Testing\Saved_models\Isolation_Forest_P-3_50"
tf_rep.export_graph(tf_model_path)

# Convert the TensorFlow model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

# Save the model to .tflite file
tflite_path = r"C:\Users\Realt\Documents\Model_Testing\Saved_models\Isolation_Forest_P-3_50.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print('TensorFlow Lite Model Conversion Complete')

# Issue suggests that the ONNX model contains an operation that is not supported by the ONNX-TensorFlow 
# conversion process. This means you might not be able to directly convert this specific ONNX model to 
# TensorFlow or TensorFlow Lite.
