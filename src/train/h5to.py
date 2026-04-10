import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('./models/fine_tuned_lstm_model.h5')

# unroll=True ile TFLite'a çevir
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 1, 11], tf.float32))

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

out_path = './models/fine_tuned_lstm_model.tflite'
with open(out_path, 'wb') as f:
    f.write(tflite_model)
print(f'Kaydedildi: {out_path} ({len(tflite_model)} bytes)')