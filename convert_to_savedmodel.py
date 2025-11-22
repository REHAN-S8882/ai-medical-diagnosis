# deployment/convert_to_savedmodel.py

import tensorflow as tf
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deployment"

H5_CANDIDATES = [
    DEPLOY / "model.h5",
    DEPLOY / "package" / "model.h5",
]

h5_path = None
for p in H5_CANDIDATES:
    if p.exists():
        h5_path = p
        break

if h5_path is None:
    print("ERROR: model.h5 not found in deployment/ or deployment/package/")
    sys.exit(1)

print("Using model:", h5_path)

saved_dir = DEPLOY / "saved_model"
if saved_dir.exists():
    print("Deleting old saved_model folder...")
    shutil.rmtree(saved_dir)

print("Loading model...")
model = tf.keras.models.load_model(str(h5_path))

print("Saving SavedModel...")
tf.saved_model.save(model, str(saved_dir))

print("SavedModel created at:", saved_dir)
