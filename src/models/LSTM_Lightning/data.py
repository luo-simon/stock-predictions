import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.misc import load_processed_dataset, split_data, create_sequences

class StockDataModule(L.LightningDataModule):
    def __init__(self, stock, feature_set, sequence_len=5, batch_size=256, permute_column=None):
        super().__init__()
        self.save_hyperparameters()

        self.stock = stock
        self.feature_set = feature_set
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.permute_column=permute_column

        self.y_scaler = None

    def setup(self, stage=None):
        # Load and split data
        df = load_processed_dataset(self.stock, start_date="20")
        X = df.drop("Close Forecast", axis=1)[self.feature_set]
        y = df["Close Forecast"]

        if self.permute_column:
            X[self.permute_column] = np.random.permutation(X[self.permute_column])

        X_train, X_val, X_test = split_data(X, verbose=False)
        y_train, y_val, y_test = split_data(y, verbose=False)

        # Normalisation
        in_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        X_train_norm = in_scaler.fit_transform(X_train.values)
        X_val_norm = in_scaler.transform(X_val.values)
        X_test_norm = in_scaler.transform(X_test.values)
        y_train_norm = self.y_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_val_norm = self.y_scaler.transform(y_val.values.reshape(-1, 1))
        y_test_norm = self.y_scaler.transform(y_test.values.reshape(-1, 1))

        # Sequencing
        X_train_seq, y_train_seq = create_sequences(X_train_norm, y_train_norm, self.sequence_len)
        X_val_seq, y_val_seq = create_sequences(X_val_norm, y_val_norm, self.sequence_len)
        X_test_seq, y_test_seq = create_sequences(X_test_norm, y_test_norm, self.sequence_len)

        # TensorDatasets
        self.train_dataset = TensorDataset(torch.tensor(X_train_seq.astype(np.float32)), torch.tensor(y_train_seq.astype(np.float32)))
        self.val_dataset = TensorDataset(torch.tensor(X_val_seq.astype(np.float32)), torch.tensor(y_val_seq.astype(np.float32)))
        self.test_dataset = TensorDataset(torch.tensor(X_test_seq.astype(np.float32)), torch.tensor(y_test_seq.astype(np.float32)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
