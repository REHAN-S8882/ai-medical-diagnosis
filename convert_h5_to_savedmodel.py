# deployment/convert_h5_to_savedmodel.py
import tensorflow as tf
from pathlib import Path

h5_path = Path("models/best.h5")
out_dir = Path("deployment/saved_model")

model = tf.keras.models.load_model(h5_path)
tf.saved_model.save(model, out_dir)
print("SavedModel written to:", out_dir.resolve())
