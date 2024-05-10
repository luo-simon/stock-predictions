# Comparative Analysis of Machine-Learning Methods to Predict Stock Price

## Repository overview
File structure adapted from *https://drivendata.github.io/cookiecutter-data-science/*
```
├── LICENSE
├── README.md          
├── data
│   ├── external       <- Data from third party sources.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── configs            <- Model configuration files
├── notebooks          <- Jupyter notebooks 
└── src                <- Source code for use in this project.
    ├── data           
    │   ├── access.py  <- Get data 
    │   └── assess.py  <- Process data and feature engineering
    ├── models         <- Folder for each model   
    │   └── example_model 
    │       ├── data.py     <- Preprocess data to suit input to model
    │       ├── evaluate.py <- Evaluate model on test set
    │       ├── model.py    <- Define model
    │       ├── train.py    <- Train model>
    │       └── tune.py     <- Tune model hyperparameters
    │   ...
    ├── tests          <- Tests, mirroring models structure
    ├── trading        <- Simulating trading strategies based on model predictions
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
```
## Libraries
- Data
    - `yfinance` to interface with Yahoo Finance datasets
    - `ta-lib` to generate technical indicators
- Machine-Learning
    - `sklearn`, `pytorch`, `statsmodels`
    - `mlflow` for logging model experiments
- Misc
    - `numpy`, `pandas`, `matplotlib`, `seaborn`, `yaml`
- Formatting/Linting
    - `black`, `flake8`
    
An experiment can be run from the command line in the following format:

```
train.py SUBCOMMAND --config CONFIG FILE
```

The subcommand can be `fit` or `tune`: the `fit` subcommand trains a model under a set of manually defined conditions specified in a configuration file, whereas the `tune` runs a specified number of trials, automatically tuning the configuration to search for the best set of hyperparameters.

The *configuration* is a YAML file specifying values for any of the experiments' attributes. Alternatively, any attributes in the YAML file can also be typed directly into the command line. An example configuration file a user might provide is shown below:

```
experiment_name: LSTM_NVDA_Experiment
stock: NVDA
features:
 - log_return
 - simple_moving_average
 - fed_funds_rate
data.sequence_len: 10
data.batch_size: 512
model.hidden_dim: 8 
model.num_layers: 2 
```


