"""Data APIs for EEGDenoiseNet dataset

Authors
 * Francesco Paissan, 2021
"""
from typing import Tuple, List
from numpy import genfromtxt, ndarray, array
import numpy as np
from scipy.stats import zscore
from random import shuffle
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from pathlib import Path
import torch


class EEGDenoiseDataset(Dataset):
    @staticmethod
    def load_samples(path: Path, label: int) -> Tuple[ndarray, ndarray]:
        X = genfromtxt(path, delimiter=",")
        y = [label] * len(X)

        return X, array(y)

    @staticmethod
    def combine_waveforms(
        clean: Tuple[ndarray, ndarray], noise: Tuple[ndarray, ndarray]
    ) -> Tuple[ndarray, ndarray]:
        """Overlaps clean signal and artifacts

        :param data: [description]
        :type data: Tuple[ndarray, ndarray]
        :return: [description]
        :rtype: Tuple[ndarray, ndarray]
        """
        clean_EEG = zscore(clean[0], axis=1)
        noise_EEG = zscore(noise[0], axis=1)

        # constructing noise vector with same dimensionality as clean data
        rep = np.ceil(len(clean_EEG) / len(noise_EEG))
        noise_EEG = np.repeat(noise_EEG, rep, axis=0)[: len(clean_EEG), :]

        return (zscore(clean_EEG + noise_EEG, axis=1), noise[1])

    def __init__(
        self, file_path: Path = Path("data/"), split: str = "", transforms: List = []
    ) -> None:
        # Labels are served as follows:
        # - 0 clean sample
        # - 1 EOG artifact on clean samples
        # - 2 EMG artifact on clean samples
        data_clean = self.load_samples(file_path.joinpath(split, "EEG_256.csv"), 0)
        data_eog = self.load_samples(file_path.joinpath(split, "EOG_256.csv"), 1)
        data_emg = self.load_samples(file_path.joinpath(split, "EMG_256.csv"), 2)

        data_eog = self.combine_waveforms(data_clean, data_eog)
        data_emg = self.combine_waveforms(data_clean, data_emg)
        data_clean = (zscore(data_clean[0], axis=1), data_clean[1])

        self.X = np.concatenate((data_eog[0], data_emg[0], data_clean[0]), axis=0)
        self.y = np.concatenate((data_eog[1], data_emg[1], data_clean[1]), axis=0)

        indices = shuffle(np.arange(len(self.X)))
        self.X = self.X[indices]
        self.y = self.y[indices]

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        return self.X[index], self.y[index]

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
            data, [int(np.ceil(len(data) * 0.3)), int(np.floor(len(data) * 0.7))]
        )

        self.train, self.val = torch.utils.data.random_split(
            train, [int(np.ceil(len(train) * 0.3)), int(np.floor(len(train) * 0.7))]
        )

    def setup(self, stage=None):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, shuffle=True, **self.dl_params)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, **self.dl_params)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, **self.dl_params)
