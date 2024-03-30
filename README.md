# Part II Project


## Components

### Data Pipeline
Collect data (access)
Preprocessing (assess)
- Filling missing values
- Validation
- Data analytics/visualization
Data Splitting  
Feature engineering
Creating batches to feed into training


### Model definitions


### Model training

### Model evaluation

### Monitoring
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