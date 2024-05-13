from typing import List, Tuple
import flwr as fl
import sys
import numpy as np
from flwr.common import Metrics
import os
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

#function to calculate the central accuracy
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)} 

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fraction_fit=1.0
        self.fraction_evaluate=0.5 
        self.evaluate_metrics_aggregation_fn = weighted_average
    
    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
         # Save aggregated_weights
        print(f"Saving round {server_round} aggregated_weights...")
        np.savez(f"round-{server_round}-weights.npz", *aggregated_weights)

        return aggregated_weights
    


strategy =SaveModelStrategy()
# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address="0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=3) ,
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
)
