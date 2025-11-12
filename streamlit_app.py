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
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gdown
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="🐶🐱 Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide"
)

# -------------------------------
# Custom CSS for Professional UI
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Poppins', sans-serif;
}
.title {
    font-size: 70px;
    font-weight: 900;
    text-align: center;
    color: #FFD700;
    text-shadow: 0px 0px 20px #FFD700;
}
.subtitle {
    font-size: 22px;
    color: #D3D3D3;
    text-align: center;
    margin-bottom: 40px;
}
.prediction {
    font-size: 65px;
    font-weight: bold;
    text-align: center;
    margin-top: 15px;
    text-shadow: 0px 0px 25px rgba(255,255,255,0.3);
}
.confidence {
    font-size: 26px;
    text-align: center;
    color: #FFDEAD;
    margin-bottom: 15px;
}
.footer {
    color: #AAAAAA;
    font-size: 15px;
    text-align: center;
    margin-top: 40px;
}
hr {
    border: 1px solid #FFD700;
}

/* -------- Upload box fix -------- */
[data-testid="stFileUploader"] section {
    background-color: #6c6d70 !important;
    border: 2px dashed #FFD700;
    border-radius: 12px;
    padding: 25px;
}
[data-testid="stFileUploader"] label {
    color: #FFD700 !important;
    font-size: 18px;
    font-weight: bold;
}
[data-testid="stFileUploader"] span {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.markdown("<h1 class='title'>🐾 Cat vs Dog Classifier 🐾</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload one or more images to instantly see whether it’s a 🐱 Cat or a 🐶 Dog!</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------
# Upload Section
# -------------------------------
uploaded_files = st.file_uploader(
    "📂 Upload cat or dog images (JPG/PNG):", type=["jpg", "png"], accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} image(s)")
    cols = st.columns(len(uploaded_files))
    for i, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file)
        img_resized = img.resize((128,128))
        img_array = img_to_array(img_resized)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0][0]
        if pred > 0.5:
            label = "🐶 This is a DOG 🐶"
            color = "#1E90FF"
            confidence = pred * 100
        else:
            label = "🐱 This is a CAT 🐱"
            color = "#FF69B4"
            confidence = (1-pred) * 100

        with cols[i]:
            st.image(img, caption=uploaded_file.name, use_column_width=True)
            st.markdown(f"<p class='prediction' style='color:{color};'>This is a {label}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='confidence'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
            st.progress(int(confidence))

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2025 Safayet Ullah | Southeast University | Made with ❤️ & Streamlit | Powered by TensorFlow 🧠</p>", unsafe_allow_html=True)
