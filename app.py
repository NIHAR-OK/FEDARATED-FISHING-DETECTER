import streamlit as st
import torch
import joblib
from model import PhishingNet

# 1. Page Configuration
st.set_page_config(page_title="Buddy's Phishing Shield", page_icon="🛡️")

# 2. Load the Files (The Brain and the Translator)
@st.cache_resource # This keeps the app fast
def load_assets():
    model = PhishingNet()
    model.load_state_dict(torch.load("phishing_model.pth", map_location=torch.device('cpu')))
    model.eval()
    vectorizer = joblib.load('DATA/vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False

# 3. The Visual Layout
st.title("🛡️ Federated Phishing Detector")
st.markdown("---")
st.write("Enter a URL below. Our AI, trained via Federated Learning, will analyze the patterns for safety.")

url_input = st.text_input("Enter URL:", placeholder="https://example.com")

if st.button("Analyze Link"):
    if not assets_loaded:
        st.error("Error: Missing 'phishing_model.pth' or 'vectorizer.pkl'. Run training first!")
    elif url_input:
        # Translate the URL
        features = vectorizer.transform([url_input]).toarray()
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Predict
        with torch.no_grad():
            prediction = model(features_tensor).item()
        
        # Display Results
        if prediction > 0.5:
            st.error(f"🛑 ALERT: Potential Phishing Scam Detected!")
            st.metric("Risk Level", f"{prediction * 100:.2f}%")
        else:
            st.success("✅ SAFE: This link looks legitimate.")
            st.metric("Safety Score", f"{(1 - prediction) * 100:.2f}%")
    else:
        st.warning("Please enter a URL first.")