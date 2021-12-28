"""
Authors
 * Francesco Paissan, 2021
"""
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.loggers import WandbLogger

from orion.client import report_objective
from modules.data_module import EEGDenoiseDM
from modules.lightning_module import BadChannelDetection
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--lr", type=float, help="Learning rate for optimization",
    )
    parser.add_argument(
        "--weight_decay", type=float, help="Weight decay for optimization",
    )

    args = vars(parser.parse_args())

    return args


def run_configuration(args, data_module):
    mod = BadChannelDetection(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss",
        dirpath="./ckp",
        filename="models-{epoch:02d}-{valid_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # wandb_logger = WandbLogger(project="EEGTagging")

    trainer = pl.Trainer(
        gpus=1, max_epochs=1, callbacks=[checkpoint_callback],  # logger=wandb_logger,
    )

    trainer.fit(model=mod, datamodule=data_module)
    t = trainer.test(datamodule=data_module, verbose=1)

    return 1 - t[0]["test/acc"]


def main():
    args = parse_arguments()
    data_module = EEGDenoiseDM()

    # Check data module. you want to opt on validation set
    error_rate = run_configuration(args, data_module)

    report_objective(error_rate)


if __name__ == "__main__":
    main()
