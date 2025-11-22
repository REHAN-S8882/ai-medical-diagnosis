import streamlit as st
import boto3
import json
import numpy as np
from PIL import Image
import io
from datetime import datetime

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =========================
# CONFIG
# =========================

ENDPOINT_NAME = "pneumonia-detector-endpoint-v7"
AWS_REGION = "ap-south-1"
IMG_SIZE = (224, 224)


# =========================
# SAGEMAKER CLIENT USING STREAMLIT SECRETS
# =========================

@st.cache_resource
def get_sagemaker_runtime():
    """
    Uses:
    - Streamlit Cloud secrets if available
    - Local AWS CLI credentials otherwise
    """

    aws_secrets = st.secrets.get("aws", None)

    if aws_secrets:
        return boto3.client(
            "sagemaker-runtime",
            region_name=aws_secrets.get("region", AWS_REGION),
            aws_access_key_id=aws_secrets["access_key_id"],
            aws_secret_access_key=aws_secrets["secret_access_key"],
        )

    return boto3.client("sagemaker-runtime", region_name=AWS_REGION)


# =========================
# PDF REPORT BUILDER
# =========================

def build_pdf_report(patient_id: str,
                     diagnosis_label: str,
                     probability: float,
                     notes: str,
                     model_name: str = "Pneumonia Detector v1") -> bytes:

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "AI-Powered Chest X-Ray Screening Report")

    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Patient / Study Info")
    c.setFont("Helvetica", 10)
    y -= 18
    c.drawString(50, y, f"Patient / Study ID: {patient_id or 'N/A'}")
    y -= 18
    c.drawString(50, y, f"Model: {model_name}")

    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "AI Assessment")
    c.setFont("Helvetica", 10)
    y -= 18
    c.drawString(50, y, f"Diagnosis: {diagnosis_label}")
    y -= 18
    c.drawString(50, y, f"Model probability (pneumonia): {probability:.3f}")

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
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return json.dumps(img_array.tolist()).encode("utf-8")


# =========================
# MODEL INVOKE
# =========================

def invoke_model(image_bytes: bytes):
    client = get_sagemaker_runtime()
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=image_bytes
    )
    result = json.loads(response["Body"].read())
    return float(result["predictions"][0][0])


# =========================
# DIAGNOSIS LOGIC
# =========================

def map_probability(prob):
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
    st.write("Upload Chest X-ray image to analyze Pneumonia probability")

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray Image (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    patient_id = st.text_input("Patient / Study ID (optional)")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        if st.button("Run AI Diagnosis"):
            with st.spinner("Analyzing using SageMaker model..."):
                payload = preprocess_image(image)
                prob = invoke_model(payload)

                diagnosis = map_probability(prob)

                st.subheader("🧾 Result")
                st.write(f"**Diagnosis:** {diagnosis}")
                st.write(f"**Probability:** {prob:.3f}")

                notes = (
                    "This AI-powered system provides decision support only. "
                    "Clinical validation by a qualified radiologist is required."
                )

                pdf_bytes = build_pdf_report(
                    patient_id=patient_id,
                    diagnosis_label=diagnosis,
                    probability=prob,
                    notes=notes
                )

                st.download_button(
                    label="📄 Download Medical PDF Report",
                    data=pdf_bytes,
                    file_name="pneumonia_report.pdf",
                    mime="application/pdf"
                )


if __name__ == "__main__":
    main()
