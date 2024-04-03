# Comparative Analysis of Machine-Learning Methods to Predict Stock Price

## Implementation Outline

### Data Access
- Decide on range, split, stock
### Feature Engineering
- Technical indicators
- Market Indices (major 5)
- Macroeconomic data (FFR)
- Fundamental Analysis Data 
- Sentiment Analysis (Difficult to do, no historical source)
### Data preprocessing (+Exploratory Data Analysis)
- Imputation
- Outliers
- Transformation (log: skewed->normal)
- Scaling (non-target variables, careful no data leakage)
### Feature selection
- Filter
    - Correlation coefficient
    - Mutual information
    - Fisher’s Score
    - Variance Threshold
    - F test
- Wrapper
    - Recursive feature elimination
    - Sequential feature selection (forward/backward)
- Embedded
    - Lasso Linear Regression
    - Random Forest Impurity
### Dimensionality reduction
- PCA
### Model selection 
- Baseline
- ARIMA
- Linear
    - Lasso (with regularisation)
- Logistic
- LSTM
    - Bidirectional LSTM (BiLSTM)
    - CNN-BiLSTM
    - TCN-BiLSTM
    - MDN-BiLSTM
    - Attention-BiLSTM
- Seq2Seq
    - Seq2Seq with Attention
    - MultiHeadAttention-BiLSTM
- Time2Vec-BiLSTM
- Time2Vec with Transformer
- Average Ensemble Model
- XGB
### Model training 
### Hyperparameter tuning
### Feature Importance // Model Interpretability
### Final models



## File structure
```
├── LICENSE
├── README.md
├── data # Data from third party sources.
├── models          # Trained and serialized models.
├── notebooks       # Jupyter notebooks.
└── src              # Source code for use in this project.
    ├── data         # Data engineering  scripts.
    │   ├── access.py # get data
        └── assess.py # clean data + feature engineering
    ├── models       # ML model engineering (a folder for each model).
    │   └── model1 
    │       ├── hyperparameters_tuning.py # hyperparameter tuning
    │       ├── model.py # model definition
    │       ├── predict.py # predict
    │       ├── preprocess.py # preprocess input data and data
    │       └── train.py #train
    └── visualization # Scripts to create exploratory and results
        │             # oriented visualizations.
        ├── evaluation.py
        └── exploration.py
```