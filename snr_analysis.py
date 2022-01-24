""" Analysis of the model performance wrt to SNR

Authors
 * Francesco Paissan, 2021
"""
from modules.lightning_module import BadChannelDetection
from modules.data_module import EEGDenoiseDM
import pytorch_lightning as pl
import numpy as np

if __name__ == "__main__":
    acc = list()
    for snr in np.arange(-7, 4, 0.5):
        dm = EEGDenoiseDM(snr_db=snr)

        model = BadChannelDetection.load_from_checkpoint(
            "ckp/models-epoch=09-valid_loss=0.00.ckpt",
            orion_args={
                "min_band_hz": 1.0,
                "lr": None,
                "weight_decay": None,
                "kernel_mult": 3.903,
            },
        )

        trainer = pl.Trainer(gpus=1)
        t = trainer.test(model, datamodule=dm)

        acc.append(t[0]["test/acc"])

    print(acc)
