import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report
""" os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  """


df = pd.read_csv("dataset/pe_file_v2.csv")
Y = df['Malware']
X = df.drop(columns = ["Malware","Name","LoaderFlags"])
X = X.iloc[:,14:]
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size =0.25)

# getDist(y_train)
from tensorflow.keras.models import load_model

# Load the existing model
model = load_model("save_5_model_of_resampling_data_downscale.h5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train,epochs=2, batch_size=1, verbose=1)
        print("Fit history : " ,model.history)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_client(
        server_address="127.0.0.1:8080", 
        client=FlowerClient().to_client(), 
        grpc_max_message_length = 1024*1024*1024
)
