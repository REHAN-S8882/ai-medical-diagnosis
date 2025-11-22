import json
import numpy as np
from PIL import Image
import boto3

# ===== CONFIG =====
ENDPOINT_NAME = "pneumonia-detector-endpoint-v7"
REGION = "ap-south-1"
IMAGE_PATH = "test_image.jpg"   # your sample X-ray
IMG_SIZE = (224, 224)           # <-- use the same size you used in training
# ===================

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0  # same normalization as training
    arr = np.expand_dims(arr, axis=0)             # shape: (1, H, W, 3)
    return arr

def main():
    # 1) Preprocess locally
    x = preprocess_image(IMAGE_PATH)
    payload = {"instances": x.tolist()}  # TF Serving format

    # 2) Call SageMaker endpoint
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = response["Body"].read().decode("utf-8")
    print("Raw response:", result)

    try:
        parsed = json.loads(result)
        print("Parsed JSON:", json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        pass

if __name__ == "__main__":
    main()
