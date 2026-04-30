import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib

# Step 1: Load the raw dataset from your folder
print("--- Step 1: Loading Dataset ---")
# Using capital DATA to match your folder structure
raw_path = 'DATA/phishing_site_urls.csv'
df = pd.read_csv(raw_path)

# Sample 30,000 rows to keep it fast on your laptop
df = df.sample(n=30000, random_state=42)

# Step 2: Clean the data and translate text to numbers
print("--- Step 2: Feature Engineering (TF-IDF) ---")
# Convert 'bad' to 1 and 'good' to 0
df['Label'] = df['Label'].map({'bad': 1, 'good': 0})

# This is your "Translator" - it picks 100 important patterns in URLs
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(df['URL']).toarray()

# Create a clean table of numbers
feature_cols = [f"feature_{i}" for i in range(100)]
processed_df = pd.DataFrame(X, columns=feature_cols)
processed_df['target'] = df['Label'].values

# Step 3: Divide the data for your 3 Clients (Companies)
print("--- Step 3: Saving Client Silos ---")
c1 = processed_df.iloc[:10000]
c2 = processed_df.iloc[10000:20000]
c3 = processed_df.iloc[20000:]

# Save these pieces into your DATA folder
c1.to_csv('DATA/client_1.csv', index=False)
c2.to_csv('DATA/client_2.csv', index=False)
c3.to_csv('DATA/client_3.csv', index=False)

# --- TASK 1: SAVE THE TRANSLATOR ---
# This saves the vectorizer so your final 'checker' script can use it
joblib.dump(vectorizer, 'DATA/vectorizer.pkl')

print("\nSuccess! Check your DATA folder. You should see:")
print("1. client_1.csv, client_2.csv, client_3.csv")
print("2. vectorizer.pkl (The Translator)")