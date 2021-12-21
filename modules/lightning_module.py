"""Training module for EMBC paper's experiments.

Authors
 * Francesco Paissan, 2021
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import recall, accuracy, f1, precision, confusion_matrix
from modules.models import ANN, ConvNet


class BadChannelDetection(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.num_classes = 3

        self.cnn = ConvNet(sr=256, sinc=True)
        self.dnn = ANN()

    def forward(self, x):
        return self.dnn(self.cnn(x))

    def loss_fn(self, out, target):
        return nn.CrossEntropyLoss()(out.view(-1, self.num_classes), target)

    def configure_optimizers(self):
        LR = 1e-2
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        return optimizer

    def _step(self, batch):
        X, y = batch

        out = self(X.unsqueeze(1).float())
        loss = self.loss_fn(out, y)

        accu = accuracy(out, y)
        prec = precision(out, y)
        rec = recall(out, y)
        f1_score = f1(out, y)

        return {"acc": accu, "loss": loss, "f1": f1_score, "prec": prec, "rec": rec}

    def training_step(self, batch, batch_idx):
        met = self._step(batch)
        # print(met)
        stage = "train"
        for key in met.keys():
            # input(f"{key}, {met[key]}")
            if key == "acc":
                self.log(f"{stage}/{key}", met[key], prog_bar=True)
            else:
                self.log(f"{stage}/{key}", met[key])

        return met["loss"]

    def validation_step(self, batch, batch_idx):
        met = self._step(batch)

        stage = "val"
        for key in met.keys():
            if key == "acc":
                self.log(f"{stage}/{key}", met[key])
            else:
                self.log(f"{stage}/{key}", met[key])

        return met["loss"]

    def test_step(self, batch, batch_idx):
        met = self._step(batch)

        stage = "test"
        for key in met.keys():
            if key == "acc":
                self.log(f"{stage}/{key}", met[key])
            else:
                self.log(f"{stage}/{key}", met[key])

        return met["loss"]
