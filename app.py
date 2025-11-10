import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configuration
IMG_SIZE = 128
MODEL_PATH = "age_gender_model.h5"

@st.cache_resource
def load_age_gender_model(path=MODEL_PATH):
    # load with compile=False to avoid deserializing training-only objects (losses/metrics)
    model = load_model(path, compile=False)
    return model

def preprocess_image(image_bytes):
    # Read image bytes to OpenCV format
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.markdown("<div class='app-title'>Age and Gender Prediction</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; margin-bottom:16px'>Upload a face image and the model will predict the age and gender.</div>", unsafe_allow_html=True)

def local_css():
    st.markdown(
        """
    <style>
    /* Brighter, friendlier palette */
    .app-title {font-size:32px; font-weight:800; color:#06b6d4; text-align:center; margin-bottom:12px}
    .uploader {border:1px solid #06b6d4; padding:12px; border-radius:10px; background:#f0fbff}
    .prediction-box {background: linear-gradient(135deg, #e0f7ff 0%, #eafff0 100%); padding:16px; border-radius:12px; box-shadow:0 8px 24px rgba(6,182,212,0.12); border:1px solid rgba(6,182,212,0.08);}
    .prediction-box .age {font-size:22px; font-weight:700; color:#0ea5e9}
    .prediction-box .gender {font-size:18px; color:#ef4444; font-weight:600}
    </style>
        """,
        unsafe_allow_html=True,
    )

local_css()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    input_img = preprocess_image(img_bytes)

    # use_container_width replaces deprecated use_column_width
    st.image(input_img[0], caption='Uploaded Image', use_container_width=True)

    with st.spinner('Loading model and predicting...'):
        model = load_age_gender_model()
        age_pred, gender_pred = model.predict(input_img)
        age = float(age_pred.flatten()[0])
        gender_prob = float(gender_pred.flatten()[0])
        gender = 'Female' if gender_prob >= 0.5 else 'Male'

        # Styled prediction box using the injected CSS
        st.markdown(
                f"""
                <div class='prediction-box'>
                    <div class='age'>Estimated age: {age:.1f} years</div>
                    <div class='gender'>Gender: {gender} ({gender_prob:.2f} probability)</div>
                </div>
                """,
                unsafe_allow_html=True,
        )
