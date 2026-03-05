import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "lung_cancer_cnn_model.keras"
FILE_ID = "18mcbs6lqFKA23kDIZyk_8R-hwhahx-Vv"



@st.cache_resource
def load_trained_model():
    # Download model if it does not exist
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    
    model = load_model("lung_cancer_cnn_model.keras", compile=False)
    return model

model = load_trained_model()

# -----------------------------
# Image preprocessing function
# -----------------------------
def preprocess_image(image):

    img = image.resize((224, 224))
    img = np.array(img)

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 AI-Based Lung Cancer Detection System")

st.write(
"""
Upload a CT scan image to check whether the model detects signs of lung cancer.
This tool uses a deep learning model trained on CT scan images.
"""
)

uploaded_file = st.file_uploader(
    "Upload a CT Scan Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded CT Scan", use_container_width=True)

    processed_img = preprocess_image(image)

    st.write("Input shape:", processed_img.shape)
    
    prediction = model.predict(processed_img)[0][0]

    st.subheader("Prediction Result")

    if prediction > 0.5:
        confidence = prediction
        st.error(f"⚠️ Cancer Detected\n\nConfidence: {prediction:.2f}")
    else:
        confidence = 1 - prediction
        st.success(f"✅ Normal\n\nConfidence: {1 - prediction:.2f}")
        
    st.progress(float(confidence))


# -----------------------------
# Disclaimer
# -----------------------------
st.markdown("---")
st.warning(
"""
⚠️ **Disclaimer:**  
This application is developed for **educational purposes only**.  
It should **not be used for real medical diagnosis**.
"""
)

