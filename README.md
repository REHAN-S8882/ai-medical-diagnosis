🩺 AI Medical Diagnosis System – Pneumonia Detection

An end-to-end AI-powered medical imaging solution that detects pneumonia from chest X-ray images using Deep Learning and cloud-based deployment. This project integrates model training, real-time inference, explainability, and an interactive web interface for clinical usage.

🚨 Problem Statement

Manual diagnosis of pneumonia from X-ray or CT scans is:

Time-consuming

Prone to human error

Dependent on expert availability

This creates delays in treatment and increases the risk of misdiagnosis.

✅ Solution Overview

This system automates the diagnosis process using Convolutional Neural Networks (CNNs) and delivers results through a user-friendly Streamlit web application.

Doctors can upload a chest X-ray and receive:

Pneumonia prediction

Confidence score

Grad-CAM heatmap visualization

Downloadable PDF medical report

🧠 AI Workflow
1. Data Collection

Medical chest X-ray images stored in:

AWS S3

2. Preprocessing

Image resizing

Normalization

Data augmentation

Noise reduction

3. Model Development

CNN with Transfer Learning

Architectures used:

ResNet

MobileNet

Framework: TensorFlow / Keras

4. Cloud Services

Training & Hosting: AWS SageMaker

Inference: SageMaker Endpoint

Storage: AWS S3

5. Deployment

Streamlit web app interface for doctors

Real-time image upload and diagnosis

6. Monitoring

Prediction confidence tracking

Error logging

Performance evaluation dashboards

🖥️ Features

✅ Real-time pneumonia detection

✅ Probability-based diagnosis logic

✅ Grad-CAM heatmap visualization

✅ Auto-generated medical PDF report

✅ Cloud deployed AI model

✅ Scalable & production-ready

📂 Project Structure
ai-medical-diagnosis/
│
├── deployment/
│   ├── app.py
│   ├── inference.py
│   ├── gradcam.py
│
├── saved_model/
├── data/
├── models/
├── test_image.jpg
└── README.md

🚀 How to Run
# Activate environment
.\venv310\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run Streamlit App
streamlit run deployment/app.py

📊 Output Example

Diagnosis: Pneumonia Detected

Probability: 92.7%

Heatmap highlights infected lung regions

Downloadable professional PDF report

📈 Deliverables

✔ AI model pipeline

✔ Streamlit interface

✔ Medical report generator

✔ Cloud-based deployment

✔ Explainable AI visualization

⚠ Disclaimer

This system is designed for educational and research purposes only. It is not a replacement for professional medical diagnosis.

👨‍💻 Developed by

Rehan Khan
AI / ML Developer
GitHub: https://github.com/REHAN-S8882
