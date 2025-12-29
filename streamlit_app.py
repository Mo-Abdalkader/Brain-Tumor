# streamlit_app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import json

# =======================
# Configuration
# =======================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
STATS_FILE = "visitor_stats.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# =======================
# Load Model
# =======================
@st.cache_resource
def load_brain_tumor_model():
    try:
        model = load_model("models/final_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_brain_tumor_model()

# =======================
# Labels & Info
# =======================
CLASS_LABELS = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary",
}

DISEASE_INFO = {
    "Glioma": {
        "description": "A tumor that occurs in the brain and spinal cord, originating from glial cells.",
        "location": "Brain and spinal cord glial cells",
        "danger": "High (can be malignant)",
        "advice": "Consult a neuro-oncologist immediately. Treatment may include surgery, radiation, and chemotherapy."
    },
    "Meningioma": {
        "description": "A tumor that arises from the meninges, the membranes surrounding the brain and spinal cord.",
        "location": "Meninges (brain and spinal cord membranes)",
        "danger": "Medium (usually benign but can grow large)",
        "advice": "Consult a neurologist. Treatment often involves monitoring or surgery depending on size and symptoms."
    },
    "Pituitary": {
        "description": "A tumor that forms in the pituitary gland, affecting hormone production.",
        "location": "Pituitary gland (base of the brain)",
        "danger": "Low to Medium (typically benign)",
        "advice": "Consult an endocrinologist. Treatment may include medication or surgery to restore hormone balance."
    },
    "No Tumor": {
        "description": "No signs of tumorous growth detected in the brain scan.",
        "location": "N/A",
        "danger": "None",
        "advice": "Continue regular check-ups. Maintain a healthy lifestyle."
    }
}

# =======================
# Visitor Statistics
# =======================
def init_stats():
    if not os.path.exists(STATS_FILE):
        stats = {"total_visits": 0, "unique_visitors": 0}
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f, indent=4)

def update_stats():
    init_stats()
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)
    stats["total_visits"] += 1
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=4)
    return stats

# =======================
# Image Preprocessing
# =======================
def preprocess_image(img: Image.Image, target_size=(224, 224)):
    # Example preprocessing: resize, grayscale, normalize
    img = ImageOps.fit(img, target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# =======================
# Streamlit Layout
# =======================
st.set_page_config(
    page_title="Brain Tumor Classification",
    layout="wide",
)

st.title("🧠 Brain Tumor Classification System")
st.write("Upload an MRI scan to detect and classify brain tumors with AI.")

# Update visitor stats
stats = update_stats()
st.sidebar.write(f"**Total Visits:** {stats['total_visits']}")

# Columns for original and processed images
col1, col_arrow, col2 = st.columns([4, 1, 4])

uploaded_file = col1.file_uploader("Upload MRI Scan", type=list(ALLOWED_EXTENSIONS))

processed_image = None
if uploaded_file:
    image_obj = Image.open(uploaded_file).convert("RGB")
    col1.image(image_obj, caption="Original Image", use_column_width=True)

    # Process image (resize, filters, etc.)
    processed_image = image_obj.resize((224, 224))
    col2.image(processed_image, caption="Processed Image", use_column_width=True)
    col_arrow.markdown("➡️", unsafe_allow_html=True)

# Prediction button
if uploaded_file and st.button("Analyze MRI Scan"):
    if model:
        st.info("Analyzing image...")
        processed_array = preprocess_image(processed_image)
        predictions = model.predict(processed_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        class_name = CLASS_LABELS.get(predicted_class, "Unknown")

        st.success(f"**Predicted Class:** {class_name} ({confidence*100:.2f}% confidence)")

        # Show detailed tumor info
        info = DISEASE_INFO.get(class_name, {})
        st.subheader("Tumor Information")
        st.write(f"**Description:** {info.get('description', 'N/A')}")
        st.write(f"**Location:** {info.get('location', 'N/A')}")
        st.write(f"**Danger Level:** {info.get('danger', 'N/A')}")
        st.write(f"**Advice:** {info.get('advice', 'N/A')}")
    else:
        st.error("Model not loaded. Cannot perform prediction.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Mohamed Abdalkader | Powered by TensorFlow & Streamlit")
