""" Analysis of the model performance wrt to SNR

Authors
 * Francesco Paissan, 2021
"""
from typing import Tuple
from scipy.io import loadmat
import numpy as np
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from modules.data_module import EEGDenoiseDataset

from modules.lightning_module import BadChannelDetection
import pytorch_lightning as pl


class Dummy(Dataset):
    def __init__(self, samples) -> None:
        self.samples = samples

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.ones(shape=self.samples[0].shape[1]),
            np.ones(shape=self.samples[0].shape[1]),
            self.samples[0][index],
            self.samples[1][index][0],
        )

    def __len__(self):
        return len(self.samples[0])


if __name__ == "__main__":
    test_set = loadmat("test_set.mat")
    clean_samples, noise_samples, label = (
        test_set["clean"],
        test_set["noise"],
        test_set["y"],
    )
    label = label.T

    model = BadChannelDetection.load_from_checkpoint(
        "ckp/models-epoch=06-valid_loss=0.00.ckpt",
        orion_args={
            "min_band_hz": 9.005,
            "lr": None,
            "weight_decay": None,
            "kernel_mult": 4.221,
        },
    )

    set_test = EEGDenoiseDataset.combine_waveforms(
        (clean_samples + 5, [0]), (noise_samples + 5, label), snr_db=0,
    )[1]

    dataset = Dummy(set_test)

    temp = dataset[0]

    trainer = pl.Trainer(gpus=1)
    trainer.test(model, test_dataloaders=DataLoader(dataset, batch_size=8))
