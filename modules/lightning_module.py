"""Training module for EMBC paper's experiments.

Authors
 * Francesco Paissan, 2021
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from modules.sincnet import ANN, ConvNet


class BadChannelDetection(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.num_classes = 3

        self.cnn = ConvNet(sr=256)
        self.dnn = ANN()

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return self.dnn(self.cnn(x))

    def loss_fn(self, out, target):
        return nn.CrossEntropyLoss()(out.view(-1, self.num_classes), target)

    def configure_optimizers(self):
        LR = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        return optimizer

    def _step(self, batch):
        X, y = batch

        out = self(X.unsqueeze(1).float())
        loss = self.loss_fn(out, y)

        logits = torch.argmax(out, dim=1)
        accu = self.accuracy(logits, y)

        return accu, loss

    def training_step(self, batch, batch_idx):
        accu, loss = self._step(batch)

        self.log("train/acc", accu, prog_bar=True)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        accu, loss = self._step(batch)

        self.log("val/loss", loss)
        self.log("val/accu", accu)

        return loss, accu

    def test_step(self, batch, batch_idx):
        accu, loss = self._step(batch)

        self.log("test/loss", loss)
        self.log("test/accu", accu)

        return loss, accu
