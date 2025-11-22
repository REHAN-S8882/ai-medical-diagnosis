import streamlit as st
import boto3
import json
import numpy as np
from PIL import Image
import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# =========================
# CONFIG
# =========================

# These fall back to sensible defaults if not set in st.secrets
ENDPOINT_NAME = st.secrets.get("ENDPOINT_NAME", "pneumonia-detector-endpoint-v7")
AWS_REGION = st.secrets.get("AWS_REGION", "ap-south-1")

# Image size used during training
IMG_SIZE = (224, 224)

# Demo images included in the repo (relative paths)
SAMPLE_IMAGES = {
    "Sample pneumonia case": "test_image.jpg",  # already in your repo root
    # You can add more later, e.g.:
    # "Sample normal chest X-ray": "sample_images/normal_demo.jpg",
}

# =========================
# SAGEMAKER CLIENT
# =========================

@st.cache_resource
def get_sagemaker_runtime():
    """
    SageMaker Runtime client.

    - Locally: uses your AWS CLI / env credentials
    - On Streamlit Cloud: uses st.secrets["aws"]
    """
    aws_secrets = st.secrets.get("aws", None)

    if aws_secrets:
        # Running on Streamlit Cloud – use secrets
        return boto3.client(
            "sagemaker-runtime",
            region_name=aws_secrets.get("region", AWS_REGION),
            aws_access_key_id=aws_secrets["access_key_id"],
            aws_secret_access_key=aws_secrets["secret_access_key"],
        )

    # Local fallback – uses your local AWS config (~/.aws/credentials)
    return boto3.client("sagemaker-runtime", region_name=AWS_REGION)


# =========================
# PDF REPORT BUILDER
# =========================

def build_pdf_report(
    patient_id: str,
    diagnosis_label: str,
    probability: float,
    notes: str,
    model_name: str = "Pneumonia Detector v1",
) -> bytes:
    """
    Build a simple 1-page PDF and return it as bytes.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Title
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "AI-Powered Chest X-Ray Screening Report")

    # Timestamp
    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(
        50,
        y,
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )

    # Patient / study info
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Patient / Study Info")
    c.setFont("Helvetica", 10)
    y -= 18
    c.drawString(50, y, f"Patient / Study ID: {patient_id or 'N/A'}")
    y -= 18
    c.drawString(50, y, f"Model: {model_name}")

    # AI assessment
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "AI Assessment")
    c.setFont("Helvetica", 10)
    y -= 18
    c.drawString(50, y, f"Diagnosis: {diagnosis_label}")
    y -= 18
    c.drawString(50, y, f"Model probability (pneumonia): {probability:.3f}")

    # Notes (wrapped)
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Notes")
    c.setFont("Helvetica", 10)
    y -= 18

    wrapped = []
    line = ""
    max_chars = 90
    for word in (notes or "").split():
        if len(line) + len(word) + 1 > max_chars:
            wrapped.append(line)
            line = word
        else:
            line += (" " if line else "") + word
    if line:
        wrapped.append(line)

    for ln in wrapped:
        c.drawString(50, y, ln)
        y -= 14
        if y < 60:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


# =========================
# IMAGE PREPROCESSING
# =========================

def preprocess_image(image: Image.Image) -> bytes:
    """
    Convert PIL image → normalized numpy array → JSON bytes
    suitable for the SageMaker TensorFlow endpoint.
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, H, W, 3)
    return json.dumps(img_array.tolist()).encode("utf-8")


# =========================
# MODEL INVOKE
# =========================

def invoke_model(image_bytes: bytes) -> float:
    """
    Call the SageMaker endpoint and return pneumonia probability (float).
    """
    client = get_sagemaker_runtime()
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=image_bytes,
    )
    result = json.loads(response["Body"].read())
    # Expecting {"predictions": [[prob]]}
    return float(result["predictions"][0][0])


# =========================
# DIAGNOSIS LOGIC
# =========================

def map_probability(prob: float) -> str:
    """
    Map probability → human-friendly diagnosis label.
    """
    if prob >= 0.75:
        return "Pneumonia Detected"
    elif prob >= 0.5:
        return "Possible Pneumonia (Moderate)"
    else:
        return "Normal"


# =========================
# STREAMLIT UI
# =========================

def main():
    st.set_page_config(page_title="AI Pneumonia Detector", layout="centered")

    st.title("🩺 AI-Powered Pneumonia Detector")
    st.write(
        "Upload a Chest X-ray image **or** use a built-in demo sample "
        "to analyze Pneumonia probability using an AWS SageMaker model."
    )

    # --- Input mode selector ---
    mode = st.radio(
        "Choose input mode",
        ["Upload image", "Use demo sample"],
        horizontal=True,
    )

    patient_id = st.text_input("Patient / Study ID (optional)")

    image = None

    if mode == "Upload image":
        uploaded_file = st.file_uploader(
            "Upload Chest X-ray Image (JPEG/PNG)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width=True)

    else:  # Use demo sample
        sample_name = st.selectbox("Choose demo X-ray", list(SAMPLE_IMAGES.keys()))
        image_path = SAMPLE_IMAGES[sample_name]

        try:
            image = Image.open(image_path)
            st.image(image, caption=f"Demo: {sample_name}", use_column_width=True)
            st.info("You are running the model in **demo mode** using a sample image.")
        except FileNotFoundError:
            st.error(f"Demo image not found at: {image_path}")
            image = None

    # --- Run model if we have an image ---
    if image is not None and st.button("Run AI Diagnosis"):
        with st.spinner("Analyzing using SageMaker model..."):
            try:
                payload = preprocess_image(image)
                prob = invoke_model(payload)
                diagnosis = map_probability(prob)

                st.subheader("🧾 Result")
                st.write(f"**Diagnosis:** {diagnosis}")
                st.write(f"**Probability (pneumonia):** {prob:.3f}")

                notes = (
                    "This AI-powered system provides decision support only. "
                    "It must not be used as a sole basis for clinical decisions. "
                    "All findings should be reviewed by a qualified radiologist."
                )

                pdf_bytes = build_pdf_report(
                    patient_id=patient_id,
                    diagnosis_label=diagnosis,
                    probability=prob,
                    notes=notes,
                )

                st.download_button(
                    label="📄 Download Medical PDF Report",
                    data=pdf_bytes,
                    file_name="pneumonia_report.pdf",
                    mime="application/pdf",
                )

            except Exception as e:
                st.error(f"Something went wrong while calling the model: {e}")


if __name__ == "__main__":
    main()
