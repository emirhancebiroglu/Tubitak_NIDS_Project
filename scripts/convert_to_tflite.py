#!/usr/bin/env python3
"""
convert_to_tflite.py v2 — Statik input shape ile LSTM → TFLite
TensorList sorununu concrete function ile çözer, Select TF Ops gerekmez.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
models_dir = ROOT / "models"

h5_path     = models_dir / "fine_tuned_lstm_model.h5"
tflite_path = models_dir / "best_lstm_model.tflite"

print(f"Yükleniyor: {h5_path}")
model = tf.keras.models.load_model(h5_path)

# Statik batch=1, timesteps=1, features=11 ile concrete function oluştur
# Bu LSTM'in TensorList dinamik shape sorununu çözer
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 1, 11], dtype=tf.float32, name="input")
])
def serving_fn(x):
    return model(x, training=False)

# Converter — concrete function üzerinden
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [serving_fn.get_concrete_function()],
    model
)

# Sadece builtin ops — Select TF Ops YOK
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.optimizations = []

print("Dönüştürülüyor...")
tflite_model = converter.convert()

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Yazıldı: {tflite_path}  ({len(tflite_model)/1024:.1f} KB)")

# Doğrulama
print("\nDoğrulama...")
interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print(f"Input  shape: {inp['shape']}  dtype: {inp['dtype']}")
print(f"Output shape: {out['shape']}  dtype: {out['dtype']}")

dummy = np.zeros([1, 1, 11], dtype=np.float32)
interp.set_tensor(inp['index'], dummy)
interp.invoke()
result = interp.get_tensor(out['index'])
print(f"Dummy inference: {result}")
print("✅ Başarılı — Select TF Ops gerektirmez.")