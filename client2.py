import flwr as fl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
""" os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  """


selected_columns = ["DllCharacteristics", "MajorImageVersion", "MajorOperatingSystemVersion"
                    ,"SizeOfStackReserve", "AddressOfEntryPoint"
                    , "Characteristics", "SizeOfHeaders", "SizeOfInitializedData"
                    , "SizeOfUninitializedData", "MinorSubsystemVersion", "CheckSum"
                    , "ImageBase", "MajorLinkerVersion", "NumberOfSections", 
                    "Subsystem", "MinorImageVersion", "SizeOfStackCommit", "e_lfanew"
                    , "e_minalloc", "e_ovno","PointerToSymbolTable", "NumberOfSymbols"
                    , "SizeOfCode", "BaseOfCode","Malware"]
                    
df = pd.read_csv("dataset/dataset_malwares_modified.csv")
df = df[selected_columns].iloc[14001:,:]
scaler = MinMaxScaler()
features = df.drop(columns=["Malware"])
label = df['Malware']
scaled_features = scaler.fit_transform(features)
# Create a DataFrame with the scaled features
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)
df = pd.concat([scaled_features_df, label.reset_index(drop=True)], axis=1)

Y = df['Malware']
X = df.drop(columns = ["Malware"])
x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size =0.25)

# Load the saved model
model = load_model("1_save_resampling_model.h5")
# Compile the loaded model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Compile the loaded model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

              

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train,epochs=10, batch_size=32, verbose=1)
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
