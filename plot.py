import numpy as np
import scipy.special
import scipy.spatial
import scipy.optimize

# must have latex suite installed in system to enable
# matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt

import glob
import sys


fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_size_inches(5, 8)
axes = [ax1, ax2]
for ax, ylabel in zip(axes, ["Regret(t)", "Est. Error"]):
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="y", scilimits=[0, 2])
    ax.grid(True)
    fig.tight_layout()
ax1.set_ylim([-1, 3000])
# ax1.set_yscale('log')
ax2.set_ylim(0, 1)

single_algs = [
    # "LCDB",
    "MaxInP",
    # "RND",
    # "MaxFirstRndNext",
    # "MaxFirstRowMaxNext",
    # "MaxFirstUCBNext2",
    "CoLSTIM",
    # "MaxFirstUCBNext",
    # "MaxFirstMaxDet",
    # "MaxDetGreedy",
    # "StaticMaxDet"
    "MaxPairUCB",
]

alg_classes = []
if len(sys.argv) >= 2:
    scale = sys.argv[1]
else:
    scale = 2
print("scale", scale)
for l in range(5, 6):
    # alg_classes.append(f"VACDB L={l} scale={scale}")
    # alg_classes.append(f"StaD L={l} scale={scale}")
    pass
for alg_cls in single_algs:
    alg_classes.append(alg_cls + f" scale={scale}")
    pass

for alg_cls in alg_classes:
    files = glob.glob(f"data/{alg_cls}/*.npz")
    print(alg_cls)
    print(len(files), files[-1])
    rd = lambda x: np.load(open(x, "rb"))
    stat = lambda dat: (dat.std(axis=0), dat.mean(axis=0))

    def draw_line(ax, dat, L=None):
        std, mean = stat(np.vstack(dat))
        std /= float(scale)
        mean /= float(scale)
        x = np.arange(0, std.shape[0])
        suffix = f" L={L}" if L is not None else ""
        ax.plot(x, mean, label=alg_cls + suffix)
        ax.fill_between(x=x, y1=mean - std, y2=mean + std, alpha=0.20)

    # print(alg_cls)
    dat = list(map(lambda x: rd(x)["r"], files))
    draw_line(ax1, dat)
    # print(dat)
    if alg_cls in ["VACDB"]:
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
fig_name = f"n=128,d=5,K=32-manyL_{scale}.png"
plt.savefig(fig_name, bbox_inches="tight")

# import os
# os.system(f"pdfcrop {fig_name} {fig_name} > /dev/null")
# os.system(f"pdffonts {fig_name} | grep 'Type 3'")
