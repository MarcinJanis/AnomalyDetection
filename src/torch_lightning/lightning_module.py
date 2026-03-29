import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

from src.models.CNNMamba import CNNMamba

class DroneClassifier(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.model = CNNMamba()
        
        metrics_task = 'binary'
        self.train_acc = Accuracy(task=metrics_task)
        self.val_acc = Accuracy(task=metrics_task)
        self.test_acc = Accuracy(task=metrics_task)
        self.test_f1 = F1Score(task=metrics_task)
        self.conf_matrix = ConfusionMatrix(task=metrics_task)
        
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y.float())
        
        self.train_acc(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y.float())
        
        self.val_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y.float())
        
        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.conf_matrix.update(logits, y)
        
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)
        self.log('test_f1', self.test_f1)
        return loss

    def on_test_epoch_end(self):
        cm = self.conf_matrix.compute()
        print(f"\nConfusion Matrix:\n{cm}")
        self.conf_matrix.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)