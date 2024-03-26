# Part II Project


## Components

### Data Pipeline
Collect data (access)
Preprocessing (assess)
Filling missing values
Validation
    Data analytics/visualization
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
├── figures         # Generated graphics and figures to be used in reporting.
└── src              # Source code for use in this project.
    ├── data         # Data engineering scripts.
    │   ├── build_features.py
    │   ├── cleaning.py
    │   ├── ingestion.py
    │   └── validation.py
    ├── models       # ML model engineering (a folder for each model).
    │   └── model1
    │       ├── dataloader.py
    │       ├── hyperparameters_tuning.py
    │       ├── model.py
    │       ├── predict.py
    │       ├── preprocessing.py
    │       └── train.py
    └── visualization # Scripts to create exploratory and results
        │             # oriented visualizations.
        ├── evaluation.py
        └── exploration.py
```