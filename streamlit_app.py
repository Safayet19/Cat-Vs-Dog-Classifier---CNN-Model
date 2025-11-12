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
# Page config
# -------------------------------
st.set_page_config(
    page_title="🐶🐱 Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Custom CSS for professional look
# -------------------------------
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { 
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Title styling */
    .main-title { 
        color: #2c3e50; 
        font-size: 3.5rem; 
        font-weight: 700; 
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    
    .subtitle { 
        color: #7f8c8d; 
        font-size: 1.3rem; 
        text-align: center; 
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card styling for images */
    .prediction-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    /* Image styling */
    .animal-image {
        width: 180px !important;
        height: 180px !important;
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 3px solid #e9ecef;
    }
    
    /* Prediction result styling */
    .prediction-result {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin: 0.5rem 0;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        width: 100%;
    }
    
    .dog-prediction {
        color: #e74c3c;
        background-color: rgba(231, 76, 60, 0.1);
    }
    
    .cat-prediction {
        color: #3498db;
        background-color: rgba(52, 152, 219, 0.1);
    }
    
    /* Confidence styling */
    .confidence-text {
        font-size: 1.4rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 0.5rem;
        color: #2c3e50;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #2ecc71;
        height: 12px;
        border-radius: 6px;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background-color: #e8f4fc;
        border: 1px dashed #3498db;
        border-radius: 8px;
    }
    
    /* Footer styling */
    .footer {
        color: #95a5a6;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #ecf0f1;
    }
    
    /* Stats container */
    .stats-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title Section
# -------------------------------
st.markdown("<h1 class='main-title'>🐾 Cat vs Dog Classifier 🐾</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload images to instantly classify them as cats or dogs with AI-powered accuracy</p>", unsafe_allow_html=True)

# -------------------------------
# Upload Section
# -------------------------------
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_files = st.file_uploader(
            "**Choose cat or dog images (JPG/PNG):**",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload one or multiple images of cats or dogs"
        )

# -------------------------------
# Results Section
# -------------------------------
if uploaded_files:
    # Stats summary
    st.markdown("---")
    st.subheader("📊 Classification Summary")
    
    # Create columns for stats
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        st.metric("Total Images", len(uploaded_files))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display images in a grid
    st.markdown("---")
    st.subheader("🔍 Predictions")
    
    # Determine grid layout based on number of images
    if len(uploaded_files) == 1:
        cols = st.columns(1)
    elif len(uploaded_files) == 2:
        cols = st.columns(2)
    elif len(uploaded_files) <= 4:
        cols = st.columns(2)
    else:
        cols = st.columns(3)
    
    # Initialize counters
    cat_count = 0
    dog_count = 0
    
    # Display each image with prediction
    for i, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file)
        img_resized = img.resize((180, 180))
        img_array = img_to_array(img_resized)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        pred = model.predict(img_array)[0][0]
        if pred > 0.5:
            label = "Dog 🐶"
            confidence = pred * 100
            color_class = "dog-prediction"
            dog_count += 1
        else:
            label = "Cat 🐱"
            confidence = (1-pred) * 100
            color_class = "cat-prediction"
            cat_count += 1

        # Display in card format
        with cols[i % len(cols)]:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.image(img_resized, use_column_width=True, output_format="auto", clamp=True)
            st.markdown(f"<div class='prediction-result {color_class}'>{label}</div>", unsafe_allow_html=True)
            st.markdown(f"<p class='confidence-text'>{confidence:.1f}% confidence</p>", unsafe_allow_html=True)
            st.progress(int(confidence))
            st.markdown(f"<small>{uploaded_file.name}</small>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Update stats
    with stat_col2:
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        st.metric("Cats Identified", cat_count)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with stat_col3:
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        st.metric("Dogs Identified", dog_count)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with stat_col4:
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        accuracy = max(cat_count, dog_count) / len(uploaded_files) * 100 if uploaded_files else 0
        st.metric("Majority", f"{accuracy:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Instructions when no files uploaded
# -------------------------------
else:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info(
            "👆 **Upload images above to get started!**\n\n"
            "This AI classifier can distinguish between cats and dogs with high accuracy. "
            "Simply upload one or multiple images and get instant predictions."
        )
        
        # Example images
        st.subheader("🎯 Examples")
        example_col1, example_col2 = st.columns(2)
        with example_col1:
            st.image("https://placekitten.com/200/200", caption="Example: Cat", use_column_width=True)
        with example_col2:
            st.image("https://placedog.net/200/200", caption="Example: Dog", use_column_width=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "<p class='footer'>"
    "© 2025 Safayet Ullah, Southeast University | "
    "Made with ❤️ & Python 🐍 | Powered by Streamlit"
    "</p>", 
    unsafe_allow_html=True
)