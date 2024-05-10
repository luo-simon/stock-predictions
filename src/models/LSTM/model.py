import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
import torch.nn as nn
import numpy as np


class LSTMModel(L.LightningModule):
    def __init__(
        self,
        sequence_len,
        input_dim,
        hidden_dim,
        num_layers,
        weight_decay,
        delta,
        output_dim=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(
            64, sequence_len, input_dim
        )  # (num_samples, seq_len, n_features)

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=False)

        self.criterion = nn.HuberLoss(delta=delta)
        self.weight_decay = weight_decay

    def setup(self, stage):
        super().setup(stage)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        out = self.fc(out[:, -1, :]).squeeze()
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=1e-8, max_lr=1, step_size_up=50
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y.squeeze())
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y.squeeze())
        self.log("val_loss", loss)
        y_pred_sign = np.sign(y_pred.cpu().detach().numpy())
        y_sign = np.sign(y.squeeze().cpu().detach().numpy())
        self.log("val_acc", np.mean(y_sign == y_pred_sign))

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y.squeeze())
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        return y_pred
