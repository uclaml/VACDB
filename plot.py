import numpy as np
import matplotlib
import scipy.special
import scipy.spatial
import scipy.optimize

# must have latex suite installed in system to enable
# matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt

import glob

files = glob.glob("data/*.npz")
print(files)
rd = lambda x: np.load(open(x, 'rb'))
# a = np.load(open(files[0], 'rb'))
regrets = list(map(lambda x: rd(x)['r'], files))
regrets = np.vstack(regrets)
std = regrets.std(axis=0)
mean = regrets.mean(axis=0)
x = np.arange(0, std.shape[0])
plt.plot(x, mean)
plt.fill_between(x=x, y1=mean-std, y2=mean+std, alpha=0.20)
plt.savefig(f"final.png")