import pandas as pd
import os
import json

from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller

# Load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv',
                 parse_dates=['date'])

data_without_date = data.drop("date", axis = 1)

print("ettm2 dataset")

save_dir = "datasets/ettm2"
method = "arima"

if not os.path.isdir(save_dir):
    os.makedirs(os.path.join(save_dir, method))

# Define the target columns (channels)
target_cols = ["OT"]

# Define the split configuration
if "etth" in save_dir:
    split_config = {
        "train": [0, 12 * 30 * 24],
        "valid": [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24],
        "test": [12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
    }
elif "covid" in save_dir:
    split_config = {
                    "train": [0, 23 * 30],
                    "valid": [24 * 30 + 3 * 30, 24 * 30 + 3 * 30 + 6],
                    "test": [
                        23 * 30,
                        23 * 30 + 4 * 30,
                    ],
                }
else:
    split_config = {
                    "train": [0, 12 * 30 * 24*4],
                    "valid": [12 * 30 * 24*4, 12 * 30 * 24 * 4 + 4 * 30 * 24*4],
                    "test": [
                        12 * 30 * 24*4 + 4 * 30 * 24*4,
                        12 * 30 * 24*4 + 8 * 30 * 24*4,
                    ],
                }


# Split the data according to the configuration
train_data = data_without_date.iloc[split_config["train"][0]:split_config["train"][1]]
valid_data = data_without_date.iloc[split_config["valid"][0]:split_config["valid"][1]]
test_data = data_without_date.iloc[split_config["test"][0]:split_config["test"][1]]

# Function to check stationarity and difference the data
def make_stationary(data, target_cols):
    differenced_data = data.copy()
    for col in target_cols:
        adf_result = adfuller(differenced_data[col])
        while adf_result[1] > 0.05:  # If p-value is greater than 0.05, the series is not stationary
            differenced_data[col] = differenced_data[col].diff().dropna()
            adf_result = adfuller(differenced_data[col].dropna())
    return differenced_data

# Make the train and test data stationary
train_data_stationary = make_stationary(train_data, target_cols)
test_data_stationary = make_stationary(test_data, target_cols)

# Function to evaluate ARIMA model for multi-channel data
def evaluate_arima_multi_channel(train, test, order, forecast_horizon, target_cols):
    history = {col: train[col].tolist() for col in target_cols}
    predictions = {col: [] for col in target_cols}
    
    for t in tqdm(range(len(test))):
        for col in target_cols:
            model = ARIMA(history[col], order=order)
            model_fit = model.fit()
            output = model_fit.forecast(steps=forecast_horizon)
            yhat = output[-1]
            predictions[col].append(yhat)
            history[col].append(test[col].iloc[t])
    
    mse = {col: mean_squared_error(test[col], predictions[col]) for col in target_cols}
    mae = {col: mean_absolute_error(test[col], predictions[col]) for col in target_cols}
    return mse, mae, predictions

# Evaluate ARIMA models
forecast_horizon = 96
order = (5, 1, 0)
mse, mae, predictions = evaluate_arima_multi_channel(train_data_stationary, test_data_stationary, order, forecast_horizon, target_cols)

# Display the results
print('MSE per channel:')
for col in mse:
    print(f'{col}: {mse[col]}')

print('\nMAE per channel:')
for col in mae:
    print(f'{col}: {mae[col]}')

# Calculate and display the aggregated MSE
aggregated_mse = sum(mse.values()) / len(mse)
print(f'\nAggregated MSE: {aggregated_mse}')

    
with open(os.path.join(save_dir, method, "mse.json"), "w") as f:
    json.dump(mse, f)
    
with open(os.path.join(save_dir, method, "mae.json"), "w") as f:
    json.dump(mae, f)
    
with open(os.path.join(save_dir, method, "preds.json"), "w") as f:
    json.dump(predictions, f)