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

    img = image.resize((244, 244))
    img = np.array(img)

    # If grayscale → convert to 3 channels
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    # If RGBA → remove alpha channel
    if img.shape[-1] == 4:
        img = img[:, :, :3]

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

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded CT Scan", use_container_width=True)

    processed_img = preprocess_image(image)

    st.write("Input shape:", processed_img.shape)
    
    pred = model.predict(processed_img)
    
    predicted_class = np.argmax(pred, axis=1)[0]
    
    confidence = np.max(pred)
    
    classes = ["cancer", "normal"]
    
    result = classes[predicted_class]
    
    st.subheader("Prediction Result")

    if result == "cancer":
    st.error(f"⚠️ Cancer Detected (Confidence: {confidence:.2f})")
    else:
        st.success(f"✅ Normal (Confidence: {confidence:.2f})")

    st.progress(confidence)


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

