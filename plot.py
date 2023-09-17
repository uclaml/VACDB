import numpy as np
import scipy.special
import scipy.spatial
import scipy.optimize

# must have latex suite installed in system to enable
# matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt

import glob

# for alg in ["vdb", "vdb-2x", "vdb-3x", "maxinp", "maxinp-2x", "maxinp-3x"]:
# for eta_scale in np.arange(0.004, 0.011, 0.001):
import sys

if len(sys.argv) > 1:
    ss = sys.argv[1]
else:
    ss = 1

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_size_inches(5, 8)
axes = [ax1, ax2]
for ax, ylabel in zip(axes, ["Regret(t)", "Est. Error"]):
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="y", scilimits=[0, 2])
    ax.grid(True)
    fig.tight_layout()
# ax1.set_ylim([-1, 10000])
# ax1.set_yscale('log')
ax2.set_ylim(0, 1)

alg_classes = [
    # "VDBGLM",
    "MaxInp",
    # "RND",
    # "MaxFirstRndNext",
    # "MaxFirstRowMaxNext",
    # "MaxFirstUCBNext2",
    # "MaxFirstUCBNext",
    # "MaxFirstMaxDet",
    # "MaxDetGreedy",
    # "StaticMaxDet"
    # "MaxPairUCB",
    "SAVE",
    # "MaxPairUCB1p6",
    # "MaxPairUCB1p5",
    # "MaxPairUCB1p1",
    # "MaxPairUCB1p3",
    # "MaxPairUCB2",
    # "MaxPairUCB4",
    # "MaxPairUCBdiv4",
    # "MaxPairUCBdiv2",
    # "MaxPairUCBd4m",
    # "MaxPairUCBd2m",
    # "MaxPairUCB1m",
    # "MaxPairUCB2m",
    # "MaxPairUCB4m",
    # "MaxPairUCBd4",
    # "MaxPairUCBd2",
    # "MaxPairUCB1",
    # "MaxPairUCB2",
    # "MaxPairUCB4",
]


for alg_cls in alg_classes:
    files = glob.glob(f"data/{alg_cls}/*.npz")
    print(files)
    rd = lambda x: np.load(open(x, "rb"))
    stat = lambda dat: (dat.std(axis=0), dat.mean(axis=0))

    def draw_line(ax, dat, L=None):
        std, mean = stat(np.vstack(dat))
        x = np.arange(0, std.shape[0])
        suffix = f" L={L}" if L is not None else ""
        ax.plot(x, mean, label=alg_cls+suffix)
        ax.fill_between(x=x, y1=mean - std, y2=mean + std, alpha=0.20)

    # print(alg_cls)
    dat = list(map(lambda x: rd(x)["r"], files))
    draw_line(ax1, dat)
    # print(dat)
    if alg_cls in ["SAVE"]:
        ax = ax2
        L = len(rd(files[0])["error"])
        for l in range(1, L):
            dat = list(map(lambda x: rd(x)["error"][l], files))
            # print(var_name, alg_cls)
            draw_line(ax2, dat, l)
    else:
        dat = list(map(lambda x: rd(x)["error"], files))
        draw_line(ax2, dat)


ax1.legend(loc="upper left", prop={"size": 7})
ax2.legend(loc="upper right", prop={"size": 7})
fig_name = f"n=128,d=5,K=32{ss}-saveL.png"
plt.savefig(fig_name, bbox_inches="tight")

# import os
# os.system(f"pdfcrop {fig_name} {fig_name} > /dev/null")
# os.system(f"pdffonts {fig_name} | grep 'Type 3'")
