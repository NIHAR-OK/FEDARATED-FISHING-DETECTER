import torch
import joblib
from model import PhishingNet

# 1. Load the "Translator" (The file you made in data_prep)
vectorizer = joblib.load('DATA/vectorizer.pkl')

# 2. Load the "Brain" (The file you made in the server)
model = PhishingNet()
model.load_state_dict(torch.load("phishing_model.pth"))
model.eval()

def check_url(url):
    print(f"\n--- Analyzing: {url} ---")
    
    # Translate the URL text into numbers the AI understands
    features = vectorizer.transform([url]).toarray()
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    # Ask the model for its opinion
    with torch.no_grad():
        prediction = model(features_tensor).item()
    
    # Print the result
    if prediction > 0.5:
        print("🛑 ALERT: This looks like a PHISHING SCAM!")
        print(f"Scam Confidence: {prediction * 100:.2f}%")
    else:
        print("✅ SAFE: This website looks legitimate.")
        print(f"Safety Confidence: {(1 - prediction) * 100:.2f}%")

# Start the interactive checker
print("--- Federated Phishing Checker is Ready ---")
while True:
    link = input("\nPaste a link to check (or type 'quit' to stop): ")
    if link.lower() == 'quit':
        break
    check_url(link)