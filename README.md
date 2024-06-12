## Installations
First of all you need to clone the original tsfm repo and install dependencies: 
```
git clone https://github.com/IBM/tsfm.git
cd tsfm
python -m venv venv
source venv/bin/activate
pip install ".[notebooks]
pip install -r ../requirements.txt
```
Then from the root you need to copy train and att_embed files:
```
cp ../train.py .
cp ../att_embed.py .
```
## Run the training process through config
To run training on you data you need to fix `config.json` file in `configs` and run this command:
```
python train.py --config ../configs/config.json
```

the datasets for training can be found by this links:

```
{
      "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
      "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
      "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
      "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
      "covid19" : "https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv"
}
```
## Run arima training
Or you can train arima model, for running this you need to execute this command under the root repo dir:
```
cd ../
python arima_training.py
```

## Citations
```
@misc{ekambaram2024tiny,
      title={Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series}, 
      author={Vijay Ekambaram and Arindam Jati and Pankaj Dayama and Sumanta Mukherjee and Nam H. Nguyen and Wesley M. Gifford and Chandra Reddy and Jayant Kalagnanam},
      year={2024},
      eprint={2401.03955},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{niu2024attention,
      title={Attention as Robust Representation for Time Series Forecasting}, 
      author={PeiSong Niu and Tian Zhou and Xue Wang and Liang Sun and Rong Jin},
      year={2024},
      eprint={2402.05370},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
