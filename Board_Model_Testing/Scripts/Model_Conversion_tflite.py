##### MODEL CONVERSION (h5 -> tflite) #####

# Keras
import keras

#Tensorflow
import tensorflow as tf


# Load the .h5 model
model_path = r"C:\Users\Realt\Documents\Model_Testing\GSTP_Saved_Model_Testing-main\Saved_models\CNNAE_F-5_30.h5"
model = tf.keras.models.load_model(model_path, compile=False)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Disable experimental lowering of tensor list ops - FOR ANY LSTM MODEL
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

# Final conversion of model
tflite_model = converter.convert()

# Save the model to .tflite file
tflite_path = r"C:\Users\Realt\Documents\Model_Testing\GSTP_Saved_Model_Testing-main\Saved_models\CNNAE_F-5_30.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print('Tensorflow Lite Model Conversion Complete')


