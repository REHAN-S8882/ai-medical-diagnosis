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

# Use the same image size you used for training
IMG_SIZE = (224, 224)

# =========================
# SAGEMAKER CLIENT
# =========================

@st.cache_resource
def get_sagemaker_runtime():
    """
    SageMaker Runtime client.

    - Locally: uses your AWS CLI / env credentials
    - On Streamlit Cloud: uses st.secrets['aws']
    """
    aws_secrets = st.secrets.get("aws", None)

    if aws_secrets:
        # Running on Streamlit Cloud – use keys from secrets
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
    c.drawString(50, y, f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

    # Notes
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Notes")
    c.setFont("Helvetica", 10)
    y -= 18

    # Simple word-wrap for notes
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
    Resize to IMG_SIZE, normalize to [0,1], make batch of 1, return JSON bytes.
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, H, W, 3)
    return json.dumps(img_array.tolist()).encode("utf-8")


# =========================
# MODEL INVOKE
# =========================

def invoke_model(image_bytes: bytes) -> float:
    """
    Call the SageMaker endpoint and return pneumonia probability as float.
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
    Map probability to human-readable label.
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
    st.write("Upload Chest X-ray image to analyze Pneumonia probability.")

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray Image (JPEG/PNG)", type=["jpg", "jpeg", "png"]
    )

    patient_id = st.text_input("Patient / Study ID (optional)")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        if st.button("Run AI Diagnosis"):
            with st.spinner("Analyzing using SageMaker model..."):
                try:
                    payload = preprocess_image(image)
                    prob = invoke_model(payload)
                    diagnosis = map_probability(prob)

                    st.subheader("🧾 Result")
                    st.write(f"**Diagnosis:** {diagnosis}")
                    st.write(f"**Probability (pneumonia):** {prob:.3f}")

                    notes = (
                        "This AI-powered system is intended for decision support only. "
                        "All findings must be reviewed and confirmed by a qualified clinician "
                        "before making any diagnostic or treatment decisions."
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
# deployment/app.py