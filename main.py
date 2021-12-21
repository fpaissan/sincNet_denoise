"""
Authors
 * Francesco Paissan, 2021
"""
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.loggers import WandbLogger

from modules.data_module import EEGDenoiseDM
from modules.lightning_module import BadChannelDetection


def main():
    mod = BadChannelDetection()
    data_module = EEGDenoiseDM()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss",
        dirpath="./ckp",
        filename="models-{epoch:02d}-{valid_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    wandb_logger = WandbLogger(project="EEGTagging")

    trainer = pl.Trainer(
        gpus=0, max_epochs=150, callbacks=[checkpoint_callback],  # logger=wandb_logger,
    )

    trainer.fit(model=mod, datamodule=data_module)
    trainer.test(datamodule=data_module, verbose=1)


if __name__ == "__main__":
    main()
