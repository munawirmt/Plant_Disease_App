import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import google.generativeai as genai

# Page Config
st.set_page_config(
    page_title="Smart Plant Disease Diagnostic",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header { font-size: 32px; font-weight: bold; color: #2E7D32; text-align: center; }
    .sub-header { font-size: 18px; color: #555; text-align: center; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌿 AI-Powered Plant Disease Diagnostic & Care System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a plant leaf image to detect diseases and get instant treatment solutions!</div>', unsafe_allow_html=True)

# 1. Model & Class Labels Load ചെയ്യൽ
@st.cache_resource
def load_ml_model():
    model = tf.keras.models.load_model('plant_disease_model.h5')
    with open('class_indices.json', 'r') as f:
        labels = json.load(f)
    return model, labels

try:
    model, labels = load_ml_model()
    st.sidebar.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error("Error loading model or json file. Please make sure 'plant_disease_model.h5' and 'class_indices.json' are in the same folder.")

# 2. Gemini API Setup (Smart Treatment Advisory-ക്ക് വേണ്ടി)
st.sidebar.subheader("🤖 AI Advisor Setup")
api_key = st.sidebar.text_input("Enter Gemini API Key (Optional)", type="password", help="Get free API key from Google AI Studio")

if api_key:
    genai.configure(api_key=api_key)

# Main App Layout (2 Columns)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Upload Plant Leaf Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("📊 Diagnostic Results")
    if uploaded_file is not None and 'model' in locals():
        if st.button("🔍 Analyze Leaf"):
            with st.spinner("Analyzing image using Deep Learning Model..."):
                # Preprocessing
                img = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                # Prediction
                predictions = model.predict(img_array)
                predicted_class_idx = str(np.argmax(predictions[0]))
                confidence = float(np.max(predictions[0])) * 100
                
                disease_name = labels.get(predicted_class_idx, "Unknown Disease")
                # Format class name nicely
                display_name = disease_name.replace("___", " - ").replace("_", " ")

                st.success(f"**Detected:** {display_name}")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

                # AI Smart Treatment Section
                st.markdown("---")
                st.subheader("💊 Smart Care & Treatment Recommendations")
                
                if api_key:
                    with st.spinner("Fetching AI Remedies & Organic Solutions..."):
                        try:
                            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                            prompt = f"The plant leaf has been diagnosed with: '{display_name}'. Provide a concise treatment plan with 3 sections: 1. Organic Solutions, 2. Chemical Treatments (if severe), 3. Prevention Tips for Farmers."
                            response = gemini_model.generate_content(prompt)
                            st.write(response.text)
                        except Exception as err:
                            st.warning("Could not fetch Gemini AI advisory. Check your API key.")
                else:
                    st.info("💡 Tip: Enter a Gemini API Key in the sidebar to get AI-generated customized treatment recommendations!")