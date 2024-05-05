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
# AUxillary methods
# def getDist(y):
#     ax = sns.countplot(y)
#     ax.set(title="Count of data classes")
#     plt.show()

# Load and compile Keras model
""" model =Sequential() """
#this model is used for the model which has smaller size of dataset
""" model.add(Dense(8, activation='relu',))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
optimizer='sgd',
metrics=['accuracy']) """


# Load dataset

df = pd.read_csv("dataset/pe_file_v1.csv")
Y = df['Malware']
X = df.drop(columns = ["Malware","Name"])
x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size =0.25)

from joblib import load

# Load the existing model
model = load('random_forest_model.joblib')

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(x_train, y_train,epochs=2, batch_size=1, verbose=1)
        hist = r.history
        print("Fit history : " ,hist)
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
