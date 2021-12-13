"""Training module for EMBC paper's experiments.

Authors
 * Francesco Paissan, 2021
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from modules.models import ANN, ConvNet, sinc


class BadChannelDetection(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.num_classes = 3

        self.cnn = ConvNet(sr=256, sinc=False)
        self.dnn = ANN()

        self.accuracy = torchmetrics.Accuracy()
        self.f1 = torchmetrics.F1(self.num_classes)
        self.rec = torchmetrics.Recall(self.num_classes)
        self.prec = torchmetrics.Precision(self.num_classes)

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
        prec = self.prec(logits, y)
        rec = self.rec(logits, y)
        f1 = self.f1(logits, y)

        return {"acc": accu, "loss": loss, "f1": f1, "prec": prec, "rec": rec}

    def training_step(self, batch, batch_idx):
        met = self._step(batch)

        stage = "train"
        for key in met.keys():
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
