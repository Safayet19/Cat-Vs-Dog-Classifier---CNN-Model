import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

import gdown
import os

# Google Drive direct download URL
MODEL_URL = "https://drive.google.com/uc?id=1AcPzYlWTCWmtU_ArX4zpQnt-GfrE1ZsC"

# Path where the model will be saved locally
MODEL_PATH = "cat_dog_model.h5"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Now you can load the model
from tensorflow.keras.models import load_model
model = load_model(MODEL_PATH)


# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="🐶🐱 Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide"
)

# -------------------------------
# CSS for design
# -------------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(to right, #FFF7E6, #FFE6F0); font-family: 'Arial'; }
.title { color: #FF5733; font-size: 50px; font-weight: bold; text-align: center; }
.footer { color: gray; font-size: 14px; text-align: center; margin-top: 20px; }
.prediction { font-size: 25px; font-weight: bold; text-align: center; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.markdown("<h1 class='title'>🐾 Cat vs Dog Classifier 🐾</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Upload one or more images to see predictions instantly!</h3>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Upload images
# -------------------------------
uploaded_files = st.file_uploader(
    "Choose cat or dog images (jpg/png):", type=["jpg","png"], accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} image(s) ✅")
    cols = st.columns(len(uploaded_files))
    for i, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file)
        img_resized = img.resize((128,128))
        img_array = img_to_array(img_resized)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        pred = model.predict(img_array)[0][0]
        if pred > 0.5:
            label = "Dog 🐶"
            confidence = pred * 100
            color = "#FF6F61"
        else:
            label = "Cat 🐱"
            confidence = (1-pred) * 100
            color = "#6FA8DC"

        with cols[i]:
            st.image(img, caption=uploaded_file.name, use_column_width=True)
            st.markdown(f"<p class='prediction' style='color:{color};'>{label}</p>", unsafe_allow_html=True)
            st.progress(int(confidence))

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2025 Safayet Ullah, Southeast University | Made with❤️ & Python 🐍 | Powered by Streamlit</p>", unsafe_allow_html=True)
