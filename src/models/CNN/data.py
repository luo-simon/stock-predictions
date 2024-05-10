import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.misc import load_processed_dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler


class CNNDataModule(L.LightningDataModule):
    def __init__(
        self,
        stock,
        feature_set,
        sequence_len,
        batch_size,
        target_var="log_return_forecast",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.stock = stock
        self.feature_set = feature_set
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.target_var = target_var

        self.setup()

    def setup(self, stage=None):
        df = load_processed_dataset(
            self.stock, start_date=f"2004-01-01", end_date="2024-01-01"
        )
        drop_cols = [c for c in df.columns if "forecast" in c.lower()]
        X = df.drop(drop_cols, axis=1)[self.feature_set]
        y = df[self.target_var].to_frame()

        # Sequencing
        X = self.process_inputs(X, self.sequence_len)
        common_index = X.index.intersection(y.index)
        X, y = X.loc[common_index], y.loc[common_index]

        # Splitting
        X_train, X_val, X_test = (
            X[:"2022-01-01"],
            X["2022-01-01":"2023-01-01"],
            X["2023-01-01":],
        )
        y_train, y_val, y_test = (
            y[:"2022-01-01"],
            y["2022-01-01":"2023-01-01"],
            y["2023-01-01":],
        )

        # Updating state
        self.X_test = X_test
        self.y_test = y_test

        # Standardisation
        input_scaler = StandardScaler()
        input_scaler.fit(X_train)
        X_train = input_scaler.transform(X_train)
        X_val = input_scaler.transform(X_val)
        X_test = input_scaler.transform(X_test)

        X_train = X_train.reshape(-1, len(self.feature_set), self.sequence_len)
        X_val = X_val.reshape(-1, len(self.feature_set), self.sequence_len)
        X_test = X_test.reshape(-1, len(self.feature_set), self.sequence_len)

        self.train_dataset = TensorDataset(
            torch.tensor(X_train).float(), torch.tensor(y_train.values).float()
        )
        self.val_dataset = TensorDataset(
            torch.tensor(X_val).float(), torch.tensor(y_val.values).float()
        )
        self.test_dataset = TensorDataset(
            torch.tensor(X_test).float(), torch.tensor(y_test.values).float()
        )

    def get_permuted_test_set(self, feature_index):
        X_test, y_test = self.test_dataset.tensors
        permuted_indices = torch.randperm(X_test.size(0))
        permuted_feature = X_test[:, feature_index, :][permuted_indices]
        X_test[:, feature_index, :] = 0
        return DataLoader(TensorDataset(X_test, y_test), batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def process_inputs(self, data: pd.DataFrame, sequence_len: int) -> pd.DataFrame:
        """
        :param data: unsequenced data
        :param sequence_len: length of the sequences to generate for each day
        :return: Pandas DataFrame where each row contains data for sequence_len previous days inclusive
        """
        if isinstance(data, pd.Series):
            data = data.to_frame()

        df = []
        for f in self.feature_set:
            feature_df = []
            for i in range(sequence_len):
                shifted_series = data[f].shift(i).rename(f"T-{i}")
                feature_df.append(shifted_series)
            feature_df = pd.concat(reversed(feature_df), axis=1).dropna()
            feature_df.columns = pd.MultiIndex.from_product([[f], feature_df.columns])
            df.append(feature_df)
        df = pd.concat(df, axis=1)
        return df
