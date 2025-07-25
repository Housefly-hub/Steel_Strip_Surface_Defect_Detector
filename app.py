import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
@st.cache_resource
def load_defect_model():
    return load_model("model/keras_model.h5")

model = load_defect_model()

# Define class labels
class_names = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scrathes", "No Defect"]

# Preprocess the uploaded image
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Set page config
st.set_page_config(page_title="Steel Defect Detector", page_icon="🧲", layout="centered")

# Custom CSS for styling
# Custom CSS for metallic background
st.markdown("""
    <style>
    body {
        background-color: #1f1f1f;
        background-image: url("https://www.transparenttextures.com/patterns/brushed-alum.png");
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: rgba(30, 30, 30, 0.9);
        color: #e5e5e5;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #d1d5db;
        text-shadow: 1px 1px 2px #999;
    }
    .subheader {
        text-align: center;
        font-size: 18px;
        color: #a0aec0;
    }
    .block-container {
        padding: 2rem;
        border-radius: 12px;
        background-color: rgba(45, 45, 45, 0.85);
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.05);
    }
    </style>
""", unsafe_allow_html=True)


# Title and subtitle
st.markdown('<div class="title">🧲 Steel Surface Defect Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload images of hot steel strip to identify surface defects</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    st.write("🔍 Analyzing the image...")
    input_image = preprocess_image(image)
    prediction = model.predict(input_image)[0]  # softmax output

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    st.subheader("🧪 Prediction Result")

    if predicted_class == "No Defect":
        st.success(f"✅ No surface defect found.\n\nThe model is {confidence:.2f}% confident.")
    else:
        st.error(f"⚠️ Defect Detected: **{predicted_class}**\n\nConfidence: {confidence:.2f}%")

    # Optional: show full class probabilities
    with st.expander("📊 View all class probabilities"):
        for i, prob in enumerate(prediction):
            st.write(f"**{class_names[i]}**: {prob*100:.2f}%")
else:
    st.info("Please upload an image to begin analysis.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Students@GEC,Rajkot| Streamlit AI App", unsafe_allow_html=True)
