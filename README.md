# 🛡️ Catching Phishers (Without Snooping on Users)

Let’s be honest: most "smart" apps are total data-hogs. Usually, to train an AI to catch phishing links, you have to send every URL you click to a giant central server. I wasn't really a fan of that "creepy" factor, so I built this project to see if we could do better.

**Live Demo:** [Try the detector here!](https://fedarated-fishing-detecter-bxwgccgmmrtc6pctu6hyaa.streamlit.app/)

---

### 🤔 The "Why" Behind This
The goal was simple: Can we make an AI smart enough to spot a scam without ever seeing your private browsing data? 

That’s where **Federated Learning** comes in. Think of it like a group project where nobody has to share their notes, but everyone still learns the answer. The model learns from your device locally, then shares only the "knowledge" (not the data) back to the main brain.

**Your data stays with you. The AI still gets smarter. Privacy wins.**

---

### 🛠️ The Secret Sauce (Tech Stack)
I kept things lean and effective:
*   **PyTorch:** The heavy lifter for the neural network.
*   **Flower (flwr):** The framework that handles the "secret handshake" between the server and the clients.
*   **Streamlit:** For the front end, because it’s clean and gets out of the way.
*   **Python:** The glue holding it all together.

---

### 📂 What's inside?
*   `app.py`: The "front door" of the project (the website).
*   `model.py`: The actual blueprint for the AI brain.
*   `server.py` & `client.py`: The logic that makes the "learning without seeing" part work.
*   `requirements.txt`: The shopping list of tools you need to run this.

---

### 🚀 Run it yourself
If you want to poke around the code on your own machine:

1.  **Clone it:** 
    `git clone https://github.com/NIHAR-OK/FEDARATED-FISHING-DETECTER.git`
2.  **Get the libraries:** 
    `pip install -r requirements.txt`
3.  **Launch:** 
    `streamlit run app.py`

---
