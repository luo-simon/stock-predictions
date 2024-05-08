# Comparative Analysis of Machine-Learning Methods to Predict Stock Price

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


