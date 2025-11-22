import io
import json
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = (224, 224)

def model_fn(model_dir):
    """Loads the Keras model inside SageMaker."""
    model = tf.keras.models.load_model(f"{model_dir}/model.h5", compile=False)
    return model

def _preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def input_fn(request_body, content_type):
    if content_type.startswith("image/") or content_type == "application/x-image":
        return _preprocess(request_body)
    raise ValueError("Unsupported content type: " + content_type)

def predict_fn(input_data, model):
    pred = model.predict(input_data, verbose=0)
    prob = float(pred[0][0])
    return prob

def output_fn(prediction, accept):
    label = "Pneumonia" if prediction >= 0.5 else "Normal"
    return json.dumps({"label": label, "probability": prediction}), "application/json"
