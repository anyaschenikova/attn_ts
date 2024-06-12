# # Clone the ibm/tsfm
# ! git clone https://github.com/IBM/tsfm.git
# # Change directory. Move inside the tsfm repo.
# %cd tsfm
# # Install the tsfm library
# !pip install ".[notebooks]"

# !pip install --upgrade einops transformers

# Standard
import os
import math
import tempfile
import json.scanner
import torch
import numpy as np
import json
import time
import fire

    
from peft import LoraConfig, PeftModel, get_peft_model, PrefixTuningConfig, TaskType
import transformers
from pathlib import Path


# Third Party
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
import numpy as np
import pandas as pd

import torch
from torch.nn.utils.rnn import pad_sequence

# First Party
from tsfm_public.models.tinytimemixer.utils import (
    count_parameters,
    plot_preds,
)

# Local
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.callbacks import TrackingCallback

from att_embed import AttentionEmbedding
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

def get_data(name, dataset_path, forecast_length, fewshot_fraction, context_length):
    timestamp_column = "date"
    if "covid" in name:
        timestamp_column = "Date"
    id_columns = []
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    if "covid" in name:
        target_columns = ["Daily_Confirmed", "Daily_Deaths"]
    if 'h' in name:
        split_config = {
                "train": [0, 12 * 30 * 24],
                "valid": [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24],
                "test": [
                    12 * 30 * 24 + 4 * 30 * 24,
                    12 * 30 * 24 + 8 * 30 * 24,
                ],
            }
    elif "covid" in name:
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
    # Understanding the split config -- slides

    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )
    
    if "covid" in name:
        data = data[data["Country/Region"]=="Russia"]
        data['Daily_Confirmed'] = data['Confirmed'].diff().fillna(0)
        data['Daily_Recovered'] = data['Recovered'].diff().fillna(0)
        data['Daily_Deaths'] = data['Deaths'].diff().fillna(0)
        data = data[["Date", "Daily_Confirmed", "Daily_Deaths"]]
        
    print("data", dataset_path)


    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "control_columns": [],
    }

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    train_dataset, valid_dataset, test_dataset = tsp.get_datasets(
        data, split_config, fewshot_fraction=fewshot_fraction, fewshot_location="first"
    )
    print(f"Data lengths: train = {len(train_dataset)}, val = {len(valid_dataset)}, test = {len(test_dataset)}")
    return train_dataset, valid_dataset, test_dataset

class TimeSeriesDataCollator:
    def __call__(self, features):
        # Assuming features is a list of dictionaries
        past_values = [f["past_values"] for f in features]
        future_values = [f["future_values"] for f in features]

        # Pad sequences
        past_values_padded = pad_sequence(past_values, batch_first=True, padding_value=0)
        future_values_padded = pad_sequence(future_values, batch_first=True, padding_value=0)

        # Return dictionary with padded sequences
        return {
            "past_values": past_values_padded,
            "future_values": future_values_padded,
            # "labels": future_values_padded,  # Assuming future values are the labels
        }

data_collator = TimeSeriesDataCollator()

def freezing_model_params(model, config):
    # Freeze the backbone of the model
    blocks = config.get("freeze")
    
    if "encoder" in blocks:
        print("Freeze encoder")
        for param in model.backbone.encoder.parameters():
            param.requires_grad = False
    if "scaler" in blocks:
        print("Freeze scaler")
    # Freeze the backbone of the model
        for param in model.backbone.scaler.parameters():
            param.requires_grad = False
    if "decoder" in blocks:
        print("Freeze decoder")
        for param in model.decoder.parameters():
            param.requires_grad = False
    if "backbone" in blocks:
        print("Freeze backbone")
        for param in model.backbone.parameters():
            param.requires_grad = False
            
def modificate_model(model, learning_type, config):

    if learning_type == "lora":
        
        lora_config = LoraConfig(
            **config.get("lora_params")
        )
        config_model = model.config
        model.decoder = get_peft_model(model.decoder, lora_config)
        model.config = config_model
        print(model)
        
    elif learning_type == "attn_embed":
        model.backbone.patching = AttentionEmbedding(
            context_window=512,
            window=96,
            c_in = 2,  # Adjust other parameters as needed
            stride=8,
            d_attn=32,
            conv_stride=24,
            n_layer=3,
            n_head=2,
            n_embd=16,
            alpha=0.1,
            initializer_range=0.2,
            embd_type='attention'
        )
        
def custom_collate_fn(batch):
    # print(f"Batch received: {batch}")
    results_batch = data_collator(batch)
    # print("result: ", results_batch)
    return results_batch
            

def main(config_path = "config.json"):
    with open("config.json") as f:
        config = json.load(f)
        
    TTM_MODEL_REVISION = "main"
    context_length = 512
    forecast_lengths = 96
    learning_type = config.get("type")
    
    print("learning_type:", learning_type)
    
    datasets = config.get("datasets")
    fewshot_fraction = config.get("fraction")

    if learning_type == "classic":
        # Load the model from HF Model Hub
        model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm/TTM", revision="main",
            head_dropout = 0.7
        )
    else:
        model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm/TTM", revision="main",
        )
        
        modificate_model(model, learning_type, config)

    print(
        "Number of params before freezing backbone",
        count_parameters(model),
    )

    freezing_model_params(model, config)

    print(
        "Number of params after freezing the backbone",
        count_parameters(model),
    )

    # Important parameters
    learning_rate = config.get("lr")
    num_epochs = config.get("num_epochs") # Ideally, we need more epochs (try offline preferably in a gpu for faster computation)
    batch_size = config.get("batch")
    
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    OUT_DIR = "ttm_attn_embed"
    
    save_name = f"b{batch_size}_lr_0{str(learning_rate).split('.')[-1]}_e{num_epochs}_{learning_type}_{time.time()}"
    save_path = os.path.join(OUT_DIR, save_name)
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f)
    
    print(f"Using learning rate = {learning_rate}")

    for name, data_url in datasets.items():
        print(name)
        train_dataset, _, test_dataset = get_data(name, data_url, forecast_lengths, fewshot_fraction, context_length)
        finetune_forecast_args = TrainingArguments(
            output_dir=save_path,
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            do_eval=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=8,
            report_to=None,
            save_strategy="epoch",
            logging_strategy="epoch",
            # save_total_limit=1,
            logging_dir=os.path.join(save_path, "logs"),  # Make sure to specify a logging directory
            load_best_model_at_end=True,  # Load the best model when training ends
            metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
            greater_is_better=False,  # For loss
            label_names=["future_values"]
        )

        # Create the early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
            early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
        )
        tracking_callback = TrackingCallback()

        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        scheduler = OneCycleLR(
            optimizer,
            learning_rate,
            epochs=num_epochs,
            steps_per_epoch=math.ceil(len(train_dataset) / (batch_size)),
        )

        finetune_forecast_trainer = Trainer(
            model=model,
            args=finetune_forecast_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            callbacks=[early_stopping_callback, tracking_callback],
            optimizers=(optimizer, scheduler),
            data_collator=custom_collate_fn
        )

        # Fine tune
        finetune_forecast_trainer.train()
        
        if config.get("save_path"):
            samples = torch.stack([test_dataset[i]["past_values"] for i in range(len(test_dataset))])
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            finetune_forecast_trainer.model = finetune_forecast_trainer.model.to(device)
            output = finetune_forecast_trainer.model(samples.to(device))
            y_hat = output.prediction_outputs.detach().cpu().numpy().tolist()
            
            if not os.path.isdir(Path(config.get("save_path")).parent):
                os.makedirs(Path(config.get("save_path")).parent)
                
            with open(config.get("save_path"), "w") as f:
                json.dump(y_hat, f)
                
        if config.get("save_actual"):
            samples = [test_dataset[i]["future_values"].tolist() for i in range(len(test_dataset))]
            
            if not os.path.isdir(Path(config.get("save_actual")).parent):
                os.makedirs(Path(config.get("save_actual")).parent)
                
            with open(config.get("save_actual"), "w") as f:
                json.dump(samples, f)

if __name__ == "__main__":
    fire.Fire(main)
