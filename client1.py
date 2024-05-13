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
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("1_save_resampling_model.h5")
# Compile the loaded model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

selected_columns = ["e_cp", "e_cparhdr", "e_maxalloc", "e_sp", "e_lfanew","Machine", 
                    "NumberOfSections", "TimeDateStamp", "NumberOfSymbols", "SizeOfOptionalHeader",
                    "Characteristics", "Magic", "MajorLinkerVersion", "MinorLinkerVersion", "SizeOfCode",
                    "SizeOfInitializedData", "SizeOfUninitializedData", "AddressOfEntryPoint", "BaseOfCode",
                    "ImageBase", "SectionAlignment", "FileAlignment", "MajorOperatingSystemVersion",
                    "MinorOperatingSystemVersion", "MajorImageVersion", "MinorImageVersion",
                    "MajorSubsystemVersion", "MinorSubsystemVersion", "SizeOfHeaders", "CheckSum", "SizeOfImage",
                    "Subsystem", "DllCharacteristics", "SizeOfStackReserve", "SizeOfStackCommit",
                    "SizeOfHeapReserve", "SizeOfHeapCommit","SuspiciousNameSection", "SuspiciousImportFunctions", "SectionsLength", "SectionMinEntropy", "SectionMinRawsize", "SectionMinVirtualsize", "SectionMaxPhysical", "SectionMaxVirtual", "SectionMaxPointerData", "SectionMaxChar", "DirectoryEntryImport", "DirectoryEntryImportSize", "DirectoryEntryExport", "ImageDirectoryEntryExport", "ImageDirectoryEntryImport"
                    ,"ImageDirectoryEntryResource", "ImageDirectoryEntryException", "ImageDirectoryEntrySecurity","Malware"]
# Load dataset
df = pd.read_csv("dataset/dataset_malwares_modified.csv")
df = df[selected_columns].iloc[7901:14000,:]
Y = df['Malware']
X = df.drop(columns = ["Malware"])
x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size =0.25)



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train,epochs=5, batch_size=50, verbose=1)       
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
