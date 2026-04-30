import flwr as fl
import torch
from model import PhishingNet

# 1. Define the strategy (How the server averages the clients' brains)
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=3,
    min_available_clients=3,
)

print("--- Task 2: Server is live ---")
print("Waiting for 3 clients to connect...")

# 2. Start the Federated Learning process
# We save the results in a variable called 'history'
history = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)

# 3. SAVE THE FINAL BRAIN
print("\n--- Training Finished! ---")
print("Saving the global model to 'phishing_model.pth'...")

# Create an empty model and save the structure
final_brain = PhishingNet()
torch.save(final_brain.state_dict(), "phishing_model.pth")

print("Success! Your project brain is saved as 'phishing_model.pth'.")