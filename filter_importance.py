""" Analysis of filter importance for classification

Authors
 * Francesco Paissan, 2021
"""
from modules.lightning_module import BadChannelDetection
import numpy as np
import torch

if __name__ == "__main__":
    model = BadChannelDetection.load_from_checkpoint(
        "ckp/models-epoch=09-valid_loss=0.00.ckpt",
        orion_args={
            "min_band_hz": 9.005,
            "lr": None,
            "weight_decay": None,
            "kernel_mult": 4.221,
        },
    )

    # frequency_filters = model.cnn.net[0].filters
    s = torch.randn_like(torch.ones(size=(16,)))
    mlp_values = model.dnn.net[0].weight

    # out = torch.matmul(mlp_values, s)
    # print(out.shape)

    filter_imp = torch.sum(mlp_values, axis=0).squeeze().detach().numpy()
    # filter_imp = filter_imp / torch.sum(filter_imp)

    with open("10.npy", "rb") as read:
        f = np.load(read).squeeze()

    # f = torch.abs(torch.fft.fft(f))
    # f = f / torch.std(f, axis=1).unsqueeze(1)

    sp = np.abs(np.fft.rfft(f, axis=1))
    freq = np.fft.rfftfreq(f.shape[-1], d=1 / 256)

    weighted_filters = sp.T * filter_imp

    import matplotlib.pyplot as plt

    # for w in weighted_filters:
    #     plt.plot(w)

    weighted_avg = np.matmul(np.expand_dims(filter_imp, 1).T, sp).squeeze() / np.sum(
        np.expand_dims(filter_imp, 1)
    )

    print(weighted_avg, freq, np.sum(np.expand_dims(filter_imp, 1)))

    plt.plot(freq, weighted_avg)
    plt.savefig("test.png")

