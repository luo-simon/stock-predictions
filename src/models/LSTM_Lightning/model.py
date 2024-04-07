import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError

class StockPricePredictor(L.LightningModule):
    def __init__(self, feature_set, sequence_len, input_dim, hidden_dim, num_layers, output_dim, lr):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(1, sequence_len , len(feature_set)) # (num_samples, seq_len, n_features)
        
        self.feature_set = feature_set
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.rmse = MeanSquaredError(squared=False)  # Set squared=False for RMSE

        self.y_scaler = None

    def on_fit_start(self):
        self.y_scaler = self.trainer.datamodule.y_scaler
        
    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # out, (hn, cn) = self.lstm(x, (h0, c0))

        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        
        # Denorm for RMSE
        y_denorm = self.y_scaler.inverse_transform(y.cpu().numpy())
        y_pred_denorm = self.y_scaler.inverse_transform(y_pred.cpu().detach().numpy())

        # Convert back to tensor for RMSE calculation
        y_denorm, y_pred_denorm = torch.tensor(y_denorm, device=self.device), torch.tensor(y_pred_denorm, device=self.device)
        
        # Calculate RMSE on denormalized values
        self.rmse(y_pred_denorm, y_denorm)

        self.log('val_loss', loss)#, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def on_validation_epoch_end(self):
        # Log the aggregated RMSE
        self.log("hp/rmse", self.rmse.compute())
        # Reset the metric for the next epoch
        self.rmse.reset()

    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

 