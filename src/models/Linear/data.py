import pandas as pd
from src.misc import load_processed_dataset, split_data

class LinearDataModule():
    def __init__(self, stock, feature_set, dataset_len):
        self.stock = stock
        self.feature_set = feature_set
        self.dataset_len = dataset_len

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.setup()

    def setup(self):
        df = load_processed_dataset(self.stock, start_date=f"{2024-self.dataset_len}-01-01", end_date="2024-01-01")
        X = df.drop("Close Forecast", axis=1)[self.feature_set]
        y = df["Close Forecast"]
        X_train, X_val, X_test = split_data(X, verbose=False)
        y_train, y_val, y_test = split_data(y, verbose=False)
        X_train = pd.concat([X_train, X_val])
        y_train = pd.concat([y_train, y_val])

        self.train_dataset = (X_train, y_train)
        self.val_dataset = (X_val, y_val)
        self.test_dataset = (X_test, y_test)
