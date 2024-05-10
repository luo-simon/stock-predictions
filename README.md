# Comparative Analysis of Machine-Learning Methods to Predict Stock Price

## Repository overview
File structure adapted from *https://drivendata.github.io/cookiecutter-data-science/*
```
├── bash                <- Bash scripts to assist in running batch experiments
├── configs             <- Configuration files for experiments
├── logs                <- Experiment logs
├── data
│   ├── external        <- Data from third party sources.
│   ├── processed       <- The final, canonical data sets for modeling.
│   └── raw             <- The original, immutable data dump.
├── notebooks           <- Jupyter notebooks 
└── src                 <- Source code for use in this project.
    ├── data           
    │   ├── access.py  <- Get data 
    │   └── assess.py  <- Process data and feature engineering
    ├── models           
    │   ├── CNN
    │   │   ├── data.py     <- Preprocess data to suit input to model
    │   │   ├── model.py    <- Define model
    │   │   └── train.py    <- Train model
    │   ...
    │   ├── statistical
    │   │   ├── ARIMA.py     
    │   │   └── ...
    │   └── custom_classes.py   <- Common overriden Lightning classes
    ├── evaluate.py             <- Evaluation, comparison and visualisation module
    ├── misc.py                 <- Miscellaneous helper functions
    └── parser.py               <- Argument parser for command line interfaces
```

Dataset and log files are not uploaded, but an example experiment log is left in `logs/`.

## Usage

First, ensure data is downloaded and processed by running `access` and `assess`.
```
src.data.access -t STOCK
src.data.assess
```

An experiment can be run from the command line, with a specified subcommand. The subcommand can be `fit` or `tune`: the `fit` subcommand trains a model under a set of manually defined conditions specified in a configuration file, whereas the `tune` runs a specified number of trials, automatically tuning the configuration to search for the best set of hyperparameters.

Statistical model experiments can be run as follows:
```
src.models.statistical.MODEL_TYPE --experiment_name EXP_NAME --stock STOCK fit
src.models.statistical.MODEL_TYPE --experiment_name EXP_NAME --stock STOCK tune 100
```

Deep learning model experiments can be run as follows:
```
src.models.MODEL_TYPE.train fit --experiment_name EXP_NAME --stock STOCK --config configs/trainer_defaults.yaml --config configs/MODEL_CONFIG.yaml
src.models.MODEL_TYPE.train tune --experiment_name EXP_NAME --stock STOCK --config configs/trainer_defaults.yaml --config configs/MODEL_CONFIG.yaml --n_trials 100
```

The *configuration* is a YAML file specifying values for any of the experiments' attributes. Alternatively, any attributes in the YAML file can also be typed directly into the command line.Example configuration files can found in `/config`.

Alternatively, see `/bash` for example helper scripts to run batches of experiments.

To evaluate, run the evaluation model:
```
src.evaluate -n EXP_NAME
```
For example, try the experiment name CNN_JPM, LSTM_NVDA, Linear_HD etc.

## Logging

To all experiments that were run, you can use `optuna-dashboard` to view the experiments in the Optuna database `optuna_studies.db`.
```optuna-dashboard sqlite://optuna_studies.db```

To view Tensorboard logs, run tensorboard, specifying the log directory.
```tensorboard --logdir logs```



## Dependencies
- Data
    - `yfinance` to interface with Yahoo Finance datasets
    - `ta-lib` to generate technical indicators
- Machine-Learning
    - `sklearn`, `statsmodels`, `pytorch(-lightning)`, `optuna`
- Misc
    - `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `yaml`
