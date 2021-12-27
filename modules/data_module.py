"""Data APIs for EEGDenoiseNet dataset

Authors
 * Francesco Paissan, 2021
"""
from pathlib import Path
from random import seed
from typing import List, Tuple

import numpy as np
import torch
from numpy import array, genfromtxt, ndarray
from pytorch_lightning import LightningDataModule
from scipy.stats import zscore
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

seed(2408)
np.random.seed(1305)


class EEGDenoiseDataset(Dataset):
    @staticmethod
    def load_samples(path: Path, label: int) -> Tuple[ndarray, ndarray]:
        X = genfromtxt(path, delimiter=",")
        y = [label] * len(X)

        return X, array(y)

    @staticmethod
    def combine_waveforms(
        clean: Tuple[ndarray, ndarray], noise: Tuple[ndarray, ndarray], snr_db: float
    ) -> Tuple[ndarray, ndarray]:
        """Combines waveforms with specified Signal to Noise ratio

        :param clean: [description]
        :type clean: Tuple[ndarray, ndarray]
        :param noise: [description]
        :type noise: Tuple[ndarray, ndarray]
        :param snr_db: [description]
        :type snr_db: float
        :return: [description]
        :rtype: Tuple[ndarray, ndarray]
        """
        rms = lambda x: np.sqrt(np.mean(x ** 2, axis=1))

        clean_EEG = clean[0]
        noise_EEG = noise[0]

        # constructing noise vector with same dimensionality as clean data
        rep = np.ceil(len(clean_EEG) / len(noise_EEG))
        noise_EEG = np.repeat(noise_EEG, rep, axis=0)[: len(clean_EEG), :]

        # Compute the mixing factor based on snr_db
        lambda_snr = rms(clean_EEG) / rms(noise_EEG) / 10 ** (snr_db / 10)
        lambda_snr = np.expand_dims(lambda_snr, 1)

        return (
            zscore(clean_EEG + lambda_snr * noise_EEG, axis=1),
            array([noise[1][0]] * len(noise_EEG)),
        )

    def __init__(
        self,
        file_path: Path = Path("data/"),
        split: str = "",
        snr_db: float = 4,
        transforms: List = [],
    ) -> None:
        # Labels are served as follows:
        # - 0 clean sample
        # - 1 EOG artifact on clean samples
        # - 2 EMG artifact on clean samples
        data_clean = self.load_samples(file_path.joinpath(split, "EEG_256.csv"), 0)
        data_eog = self.load_samples(file_path.joinpath(split, "EOG_256.csv"), 1)
        data_emg = self.load_samples(file_path.joinpath(split, "EMG_256.csv"), 2)

        data_eog = self.combine_waveforms(data_clean, data_eog, snr_db)
        data_emg = self.combine_waveforms(data_clean, data_emg, snr_db)
        data_clean = (zscore(data_clean[0], axis=1), data_clean[1])

        self.X = np.concatenate((data_eog[0], data_emg[0], data_clean[0]), axis=0)
        self.y = np.concatenate((data_eog[1], data_emg[1], data_clean[1]), axis=0)
        self.clean_samples = np.concatenate(
            (data_clean[0], data_clean[0], data_clean[0]), axis=0
        )

    def __getitem__(self, index: int) -> Tuple[ndarray, ndarray]:
        return self.clean_samples[index], self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class EEGDenoiseDM(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        file_path: Path = Path("data"),
        transforms: List = [],
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.transforms = transforms

        self.dl_params = {
            "batch_size": batch_size,
            "num_workers": 4,
            "persistent_workers": True,
            "pin_memory": False,
        }

        data = EEGDenoiseDataset(file_path)
        train, self.test = torch.utils.data.random_split(
            data,
            [int(np.floor(len(data) * 0.75)), int(np.ceil(len(data) * 0.25))],
            generator=torch.Generator().manual_seed(1305),
        )

        self.train, self.val = torch.utils.data.random_split(
            train,
            [int(np.floor(len(train) * 0.7)), int(np.ceil(len(train) * 0.3))],
            generator=torch.Generator().manual_seed(2408),
        )

    def setup(self, stage=None):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, shuffle=True, **self.dl_params)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, **self.dl_params)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, **self.dl_params)


# if __name__ == "__main__":
#     a = EEGDenoiseDataset()

#     import matplotlib.pyplot as plt

#     c, X, y = a[0]

#     sp = np.fft.rfft(c)
#     freq = np.fft.rfftfreq(c.shape[-1], d=1 / 256)
#     plt.plot(freq, np.abs(sp))

#     sp = np.fft.rfft(X)
#     freq = np.fft.rfftfreq(X.shape[-1], d=1 / 256)
#     plt.plot(freq, np.abs(sp))

#     plt.savefig("test_data.png")

# a = EEGDenoiseDM()

# X, y = [], []
# for s in a.train:
#     X.append(s[0])
#     y.append(s[1])

# X = array(X)
# y = array(y)
# print(y.shape)

# from scipy.io import savemat

# savemat("train_set.mat", {"X": X, "y": y})

# from scipy.io import loadmat

# data = loadmat("val_set.mat")
# print(data["X"].shape, data["y"].T.shape)
