"""Model definition for models proposed in EMBC paper.

Authors
 * Francesco Paissan, 2021
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.core.fromnumeric import shape
from torch.autograd import Variable
from torch.nn.modules.conv import Conv1d


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :,
        getattr(
            torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda]
        )().long(),
        :,
    ]
    return x.view(xsize)


def sinc(band, t_right):
    y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = flip(y_right, 0)

    y = torch.cat([y_left, Variable(torch.ones(1)).cuda(), y_right])

    return y


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158

    Francesco Paissan: This code comes from the original repo.
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=256,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        min_low_hz=0,
        min_band_hz=1,
    ):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = (
                "SincConv only support one input channel (here, in_channels = {%i})"
                % (in_channels)
            )
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(
            self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
        )
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = (
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        )  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class ConvNet(nn.Module):
    def __init__(
        self, sr, sinc=True, min_band_hz=None, kernel_mult=None,
    ):
        super().__init__()
        if not sinc:
            self.net = nn.Sequential(
                nn.Conv1d(
                    1, 16, kernel_size=int(np.ceil(sr / kernel_mult)), padding="same",
                ),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
        else:
            self.net = nn.Sequential(
                SincConv_fast(
                    16,
                    kernel_size=int(np.ceil(sr / kernel_mult)),
                    sample_rate=sr,
                    padding="same",
                    min_low_hz=0,
                    min_band_hz=min_band_hz,
                ),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                # nn.MaxPool1d(kernel_size=25),
                # nn.Conv1d(16, 32, kernel_size=3),
                # nn.BatchNorm1d(32),
                # nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )

    def forward(self, X):
        y_hat = X
        for idx, layer in enumerate(self.net):
            if idx == 0:
                filt = layer(y_hat)
            else:
                y_hat = layer(filt)

        return filt, self.net(X)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3),)

    def forward(self, X):
        return self.net(X)


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    f = ConvNet(sr=256)
    g = ANN()

    X = torch.randn(size=(1, 1, 512))
    print(f(X)[0].shape, f(X)[1].shape)
