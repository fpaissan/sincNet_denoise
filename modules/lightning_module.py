"""Training module for EMBC paper's experiments.

Authors
 * Francesco Paissan, 2021
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import recall, accuracy, f1, precision, confusion_matrix
from modules.models import ANN, ConvNet
from numpy import save


class BadChannelDetection(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.num_classes = 3
        self.sinc = True

        self.cnn = ConvNet(sr=256, sinc=self.sinc)
        self.dnn = ANN()

        self.automatic_optimization = False

    def forward(self, x):
        h = self.cnn(x)
        return torch.mean(h[0], axis=1).squeeze(), self.dnn(h[1])

    def loss_fn(self, y_hat, target, signal, alpha):
        ce = nn.CrossEntropyLoss()(y_hat.view(-1, self.num_classes), target)
        den_component = torch.mean(torch.sqrt((signal[1] - signal[0]) ** 2))

        return (1 - alpha) * den_component + alpha * ce

    def configure_optimizers(self):
        LR = 1e-3
        optimizer_dnn = torch.optim.Adam(self.dnn.parameters(), lr=LR)
        optimizer_cnn = torch.optim.Adam(self.cnn.parameters(), lr=LR * 10)

        return [optimizer_dnn, optimizer_cnn]

    def _step(self, batch, infer="dnn"):
        clean, noise, X, y = batch

        filtered_signal, y_hat = self(X.unsqueeze(1).float())
        den_samples = (clean, filtered_signal)

        if infer == "dnn":
            alpha = 0.6
        else:
            alpha = 0.4

        alpha = 1
        loss = self.loss_fn(y_hat, y, den_samples, alpha)

        accu = accuracy(y_hat, y)
        prec = precision(y_hat, y)
        rec = recall(y_hat, y)
        f1_score = f1(y_hat, y)

        return {"acc": accu, "loss": loss, "f1": f1_score, "prec": prec, "rec": rec}

    def training_step(self, batch, batch_idx):
        dnn_opt, cnn_opt = self.optimizers()

        # Update CNN filters
        met = self._step(batch, infer="cnn")
        cnn_opt.zero_grad()
        self.manual_backward(met["loss"])
        cnn_opt.step()

        # Update DNN filters
        met = self._step(batch, infer="dnn")
        dnn_opt.zero_grad()
        self.manual_backward(met["loss"])
        dnn_opt.step()

        stage = "train"
        for key in met.keys():
            # input(f"{key}, {met[key]}")
            if key == "acc" or key == "loss":
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

    def on_train_epoch_end(self, trainer=None, pl_module=None):
        if self.sinc:
            first_conv_filter = self.cnn.net[0].filters.clone()
            first_conv_filter = first_conv_filter.detach().cpu().numpy()

            with open(f"viz/{self.current_epoch}.npy", "wb") as write:
                save(write, first_conv_filter)
