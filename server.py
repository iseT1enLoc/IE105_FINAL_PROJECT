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


# # Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,    
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
) 
# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address="0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=2) ,
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
)
