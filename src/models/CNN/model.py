import lightning as L
import torch
import torch.nn as nn
import numpy as np

class CNNModel(L.LightningModule):
    def __init__(self, sequence_len, in_channels, out_channels, kernel_size, weight_decay, delta):
        super().__init__()
        assert sequence_len >= kernel_size
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(64, in_channels, sequence_len) # (N, C, L)

        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size) # (batch_size, in_channels, in_seq_len) -> (batch_size, , out_channels, out_seq_len)
        self.pool = nn.MaxPool1d(sequence_len-kernel_size+1) # (N, C, L) -> (N, C, 1)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(in_features=out_channels, out_features=1)
    
        self.criterion = nn.HuberLoss(delta=delta)
        self.weight_decay = weight_decay

    def setup(self, stage):
        super().setup(stage)
        
    def forward(self, x):
        x = self.cnn(x)
        x = self.pool(x).squeeze(1)
        x = self.softmax(x).squeeze(2)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=1, step_size_up=50)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss)

        y_pred_sign = np.sign(y_pred.cpu().detach().numpy().squeeze())
        y_sign = np.sign(y.cpu().detach().numpy().squeeze())
        self.log('val_acc', np.mean(y_sign == y_pred_sign))

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('test_loss', loss)
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        return y_pred
