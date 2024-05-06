import flwr as fl
import sys
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {server_round} aggregated_weights...")
            np.savez(f"round-{server_round}-weights.npz", *aggregated_weights)
            
        return aggregated_weights


# Create strategy and run server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address="0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=3) ,
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
)
