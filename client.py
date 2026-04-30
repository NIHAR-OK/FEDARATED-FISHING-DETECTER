
import torch
import pandas as pd
import flwr as fl
import argparse
from model import PhishingNet, get_parameters, set_parameters

# --- SAFETY CHANGE 1: Use CPU to avoid GPU compatibility errors ---
device = torch.device("cpu")

# 1. Ask the computer which "Student ID" this terminal is (1, 2, or 3)
parser = argparse.ArgumentParser(description="Flower Client")
parser.add_argument("--id", type=int, required=True)
args = parser.parse_args()

# 2. Load the private data from your DATA folder
print(f"--- Client {args.id} is loading private data ---")
data = pd.read_csv(f'DATA/client_{args.id}.csv')
X = torch.tensor(data.drop('target', axis=1).values, dtype=torch.float32).to(device)
y = torch.tensor(data['target'].values, dtype=torch.float32).unsqueeze(1).to(device)

# 3. Build the local model "brain"
net = PhishingNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

# 4. Define how the client behaves (Training and Testing)
class PhishingClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_parameters(net)

    def fit(self, parameters, config):
        set_parameters(net, parameters)
        # Train locally for 2 rounds
        for _ in range(2):
            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print(f"Done! Client {args.id} finished its local lesson.")
        return get_parameters(net), len(X), {}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        with torch.no_grad():
            outputs = net(X)
            # Check how many it got right
            accuracy = ((outputs > 0.5).float() == y).float().mean()
        return 0.0, len(X), {"accuracy": float(accuracy)}

# --- SAFETY CHANGE 2: Updated connection code to remove yellow warnings ---
print(f"--- Client {args.id} is connecting to the teacher (Server) ---")
fl.client.start_client(
    server_address="127.0.0.1:8080", 
    client=PhishingClient().to_client()
)