import matplotlib.pyplot as plt
import numpy as np

fsize = 15
tsize = 18
tdir = "in"
major = 5.0
minor = 3.0
lwidth = 0.8
lhandle = 2.0
plt.style.use("default")
# plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = fsize
plt.rcParams["legend.fontsize"] = tsize
plt.rcParams["xtick.direction"] = tdir
plt.rcParams["ytick.direction"] = tdir
plt.rcParams["xtick.major.size"] = major
plt.rcParams["xtick.minor.size"] = minor
plt.rcParams["ytick.major.size"] = 5.0
plt.rcParams["ytick.minor.size"] = 3.0
plt.rcParams["axes.linewidth"] = lwidth
plt.rcParams["legend.handlelength"] = lhandle

if __name__ == "__main__":
    x = np.arange(-7, 4, 0.5)
    y = [
        0.9849379658699036,
        0.9849379658699036,
        0.9849379658699036,
        0.9849379658699036,
        0.9849379658699036,
        0.9855286478996277,
        0.9855286478996277,
        0.9855286478996277,
        0.9855286478996277,
        0.9855286478996277,
        0.9855286478996277,
        0.9855286478996277,
        0.9855286478996277,
        0.9852333068847656,
        0.9793266654014587,
        0.9613112807273865,
        0.9338452219963074,
        0.8948612213134766,
        0.8505611419677734,
        0.8041937351226807,
        0.7634376883506775,
        0.7235676050186157,
    ]

    plt.title("Classification accuracy wrt to SNR")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Test accuracy")
    plt.plot(x, y, ".")
    plt.savefig("test.png")
