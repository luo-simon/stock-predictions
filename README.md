# Comparative Analysis of Machine-Learning Methods to Predict Stock Price

## Implementation Outline

### Data Access
- Data source: Yahoo Finance via `yfinance` API
- Columns: `[Open, High, Low, Close, Volume]`
- 5 year (2018-2023), 80/10/10 split.
### Feature Engineering
- [x] Lagged variables
- [x] Transformed (logged, % change)
- [x] Date features (dayofweek, weekofyear)
- [x] Technical indicators
- [x] Macroecomic indicators
    - Market indices (SPX, DJI, IXIC, N225, FTSE)
    - Fed Funds Rates
- [ ] Sentiment analysis: couldn't find historical news data
- [ ] Fundamental analysis features: difficult automate: locate/parse company financials
### Data preprocessing
- [x] Imputation, outliers
- [x] Scaling (non-target variables, careful no data leakage) where necessary
### Feature selection
- Filter methods
    - [x] Correlation coefficient
    - [x] Mutual information
    - [ ] Fisher’s Score
- Wrapper
    - [x] Recursive feature elimination
    - [x] Sequential feature selection (forward/backward)
- Embedded
    - [x] LASSO Regularisation
    - [x] Random Forest Importance
### Dimensionality reduction
- [x] PCA
### Model selection 
- [x] Baseline
- [x] ARIMA
    - [ ] ARIMAX
- [x] Linear
    - [x] Lasso L1
    - [x] Ridge L2
- [x] LSTM
- [x] XGBoost
- [x] Transformer
- [ ] *Effect of encoder/decoder/attention layers*
### Model training 
### Hyperparameter tuning
### Feature Importance // Model Interpretability
- [ ] Global Model-Agnostic Methods
    - [ ] Permutation Feature Importance
- [ ] Local Model-Agnostic Methods
    - [ ] Local Surrogate (LIME)
    - [ ] SHAP (SHapley Additive exPlanations)
### Trading Simulation

## Evaluation
- [ ] Summary table of results (average evaluation metrics, PnL)
    - [ ] Average across 10 stocks
    - [ ] Interpretation of results
- [ ] Visualisation of predictions
- [ ] Compare different models and discuss pros/cons

## Remaining Tasks
- [ ] Feature Importance // Model Interpretability
- [ ] Collate results + produce visualisations + compare
- [ ] Code clean up

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
├── figures            <- Generated graphics and figures to be used in reporting
├── notebooks          <- Jupyter notebooks 
├── requirements.txt   <- The requirements file for reproducing the analysis environment
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