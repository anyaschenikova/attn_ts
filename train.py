import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from modeling_tinytimemixer import TinyTimeMixerConfig, TinyTimeMixerModel

import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# Step 1: Download the Datasets
dataset_urls = {
    "Australian Electricity Demand": "https://zenodo.org/record/4659727/files/electricity.csv",
    "Australian Weather": "https://zenodo.org/record/4654822/files/weatherAUS.csv",
    "Bitcoin dataset": "https://zenodo.org/record/5122101/files/BitcoinHeistData.csv",
    "KDD Cup 2018 dataset": "https://zenodo.org/record/4656756/files/Metro_Interstate_Traffic_Volume.csv",
    "London Smart Meters": "https://zenodo.org/record/4656091/files/halfhourly_dataset.csv",
    "Saugeen River Flow": "https://zenodo.org/record/4656058/files/SaugeenRiverFlowData.csv",
    "Solar Power": "https://zenodo.org/record/4656027/files/SolarPrediction.csv",
    "Sunspots": "https://zenodo.org/record/4654722/files/Sunspots.csv",
    "Solar": "https://zenodo.org/record/4656144/files/solar_AL.csv",
    "US Births": "https://zenodo.org/record/4656049/files/US_births_2000-2014_SSA.csv",
    "Wind Farms Production data": "https://zenodo.org/record/4654858/files/WindFarmProduction.csv",
    "Wind Power": "https://zenodo.org/record/4656032/files/WindPrediction.csv"
}

os.makedirs("datasets", exist_ok=True)

for name, url in dataset_urls.items():
    response = requests.get(url)
    with open(os.path.join("datasets", name + ".csv"), 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {name}")

# Step 2: Load and Preprocess the Data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_and_preprocess_data(file_path, seq_length=64):
    df = pd.read_csv(file_path)
    
    data = df.values[:, -1]

    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

    return train_data, test_data, scaler

def create_dataloader(data, seq_length=64, batch_size=3000):
    dataset = TimeSeriesDataset(data, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Step 3: Prepare the DataLoader for Training
seq_length = 64
batch_size = 3000

train_loaders = {}
test_loaders = {}

for name in dataset_urls.keys():
    file_path = os.path.join("datasets", name + ".csv")
    train_data, test_data, scaler = load_and_preprocess_data(file_path, seq_length)
    
    train_loader = create_dataloader(train_data, seq_length, batch_size)
    test_loader = create_dataloader(test_data, seq_length, batch_size)
    
    train_loaders[name] = train_loader
    test_loaders[name] = test_loader

print("DataLoaders created for all datasets.")

config = TinyTimeMixerConfig()

model = TinyTimeMixerModel(config)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_loader:
        inputs, targets = batch  # Adjust this line based on your dataset structure
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
