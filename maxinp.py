import numpy as np
import matplotlib
import scipy.special
import scipy.spatial
import scipy.optimize

# must have latex suite installed in system to enable
# matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt

import sys

if len(sys.argv) == 2:
    seed = int(sys.argv[1])
else:
    seed = 27
print("seed", seed)
rng = np.random.default_rng(seed)
mu = scipy.special.expit
dtype = np.float64

# TODO: 50000
T = 2500
delta = 1.0 / T / 10
d = 10
sigma_0 = 1.0 / (2**2)
L = int(np.ceil(np.log2(1.0 / 2 / sigma_0)))
lmbda = 0.001
kappa = 0.1
L_mu = 0.25
M_mu = 0.25
K = 50

theta_star = rng.random((d,), dtype=dtype)
theta_star = theta_star / np.sqrt(theta_star @ theta_star)
theta_star *= 2
theta_star *= 3
x_orig = rng.integers(0, 2**d, size=(K,), dtype=np.int32)
cA = ((x_orig.reshape(-1, 1) & (2 ** np.arange(d))) != 0) * 2.0 - 1.0
cA += rng.random(cA.shape) - 1
# cA = rng.random((K, d)) - 1


class VDBGLM:
    def __init__(self) -> None:
        self.theta = np.ones((L + 1, d)) * 0.5
        self.V = np.zeros((L + 1, d, d), dtype=dtype) + np.eye(d) * lmbda
        self.Vinv = (
            1.0 / lmbda * np.zeros((L + 1, d, d), dtype=dtype) + np.eye(d) / lmbda
        )
        self.xy_diff = [np.zeros((0, d), dtype=dtype) for _ in range(L + 1)]
        self.r = [np.array([], dtype=dtype) for _ in range(L + 1)]
        self.R = np.zeros((T + 1), dtype=dtype)
        self.sigma_true = np.zeros((T + 1))
        self.sigma_bar = np.zeros((T + 1))

    def MLE(self, l: int) -> None:
        theta_0 = self.theta[l]
        func = lambda theta: (
            kappa * lmbda * theta @ theta
            + (
                np.log(np.exp(self.xy_diff[l] @ theta) + 1)
                - self.r[l].reshape(-1, 1) * self.xy_diff[l] @ theta
            ).sum(axis=0)
        )
        grad = lambda theta: (
            # 2 * kappa * lmbda * theta
            + (
                (mu(self.xy_diff[l] @ theta) - self.r[l]).reshape(-1, 1)
                * self.xy_diff[l]
            )
            .sum(axis=0)
            .flatten()
        )

        # gradient descent
        for _ in range(50):
            step = grad(theta_0)
            # print(step)
            theta_0 -= 0.0001 * step
        self.theta[l] = theta_0
        # print(theta_0, "hat")
        # print(theta_star, "*")

        # optimizer
        # res = scipy.optimize.minimize(
        #     func,
        #     theta_0,
        #     jac=grad,
        #     method="BFGS",
        #     options={"disp": False, "gtol": 1e-04},
        # )
        # self.theta[l] = res.x
        # print(res.x)
        return None

    def main(self) -> None:
        u = mu(cA @ theta_star)
        x_star_idx = np.argmax(u)

        g_xy_diff = cA.reshape(K, 1, d) - cA.reshape(1, K, d)
        g_xy_diff_outer = g_xy_diff.reshape(K, K, d, 1) @ g_xy_diff.reshape(K, K, 1, d)
        p = mu(g_xy_diff @ theta_star)

        # p_inc_idx = np.argsort(np.abs(p - 0.5), axis=None)
        # ppp = np.sort(np.abs(p - 0.5), axis=None)
        # p_inc_pair = np.unravel_index(p_inc_idx, (K, K))
        # zz = list(zip(*p_inc_pair))
        # print(zz[:K])
        # print(zz[K+1])
        # print("p", p)
        # print("psi", cA)
        # print("theta_star", theta_star)

        var = np.ones((L, K, K), dtype=dtype)
        cPsi = np.zeros((L + 1))
        for l in range(L):
            var[l] = np.sqrt(
                (g_xy_diff.reshape(K, K, 1, d) @ self.Vinv[l])
                @ g_xy_diff.reshape(K, K, d, 1)
            ).reshape(K, K)
        D = np.ones((L, K), dtype=dtype)
        for t in range(1, 1 + T):
            Dt = D.sum(axis=0) == L
            assert np.any(Dt)

            mask = Dt.reshape(-1, 1) * Dt.reshape(1, -1)
            # mask = mask.astype(dtype) + np.eye(K)
            Lvar = var + 1
            sel = mask * Lvar.min(axis=0)

            x_i, y_i = np.unravel_index(
                np.argmax(mask * Lvar.min(axis=0), axis=None), (K, K)
            )
            
            # mask = Dt.reshape(-1, 1) * Dt.reshape(1, -1)
            # mask = mask.astype(dtype) - np.eye(K)
            # Lvar = var - 1000
            # sel = mask * Lvar.min(axis=0)
            # x_i, y_i = np.unravel_index(
            #     np.argmin(mask * Lvar.min(axis=0), axis=None), (K, K)
            # )

            # x_i = p_inc_pair[0][-t % d * 2 * 2]
            # y_i = p_inc_pair[1][-t % d * 2 * 2]
            # ()

            # x_i, y_i = np.argsort(cA @ self.theta[0])[-2:]

            if (x_i, y_i) == (20, 20):
                pass
            # x_i = rng.integers(0, K)
            # y_i = rng.integers(0, K)
            x = cA[x_i]
            y = cA[y_i]
            xy_diff = x - y
            r = rng.binomial(1, p[x_i, y_i])
            self.R[t] = (2 * cA[x_star_idx] - x - y) @ theta_star + self.R[t - 1]
            # self.R[t] = np.linalg.norm(self.theta[0] - theta_star)

            # l = np.ceil(np.log2(bar_sigma_t / sigma_0)).astype(np.int32)[0]
            # l = np.clip(l, 1, L) - 1
            l = 0
            # l = rng.integers(0, L)
            # print(x_i, y_i)
            self.V[l] += g_xy_diff_outer[x_i, y_i]
            self.Vinv[l] = np.linalg.inv(self.V[l])
            self.xy_diff[l] = np.vstack([self.xy_diff[l], xy_diff])
            self.r[l] = np.append(self.r[l], r)
            cPsi[l] += 1
            self.MLE(l)

            # calculation in Saha's paper
            eta_tl = (
                1.0 / kappa * np.sqrt(d / 2 * np.log(1 + 2 * t / d) + np.log(1 / delta))
            )
            eta_tl *= 0.10
            var[l] = np.sqrt(
                (g_xy_diff.reshape(K, K, 1, d) @ self.Vinv[l])
                @ g_xy_diff.reshape(K, K, d, 1)
            ).reshape(K, K)
            ucb = g_xy_diff @ self.theta[l] + eta_tl * var[l]
            # print(ucb, eta_tl)
            cond = ucb.reshape(K, K) >= 0
            D[l] = cond.sum(axis=1) == K

            print(f"{t}", end="\r")

        print(x_star_idx, "x_star")
        return cPsi


v = VDBGLM()
Psi = v.main()
v.MLE(0)
v.MLE(1)
np.set_printoptions(precision=6, suppress=True)
print("est", v.theta)
print("----------")
print("gt", theta_star)
print("layer sample", Psi)
print(np.linalg.norm(v.theta - theta_star, axis=1))

# plt.plot(v.sigma_bar[1:] - v.sigma_true[1:])
# plt.show()
# plt.savefig("sigma-bar-true-005.png")

plt.plot(v.R[1:])
np.savez(f"data/maxinp-3x/R{seed:03d}", r=v.R)
# plt.show()
plt.savefig(f"data/maxinp-3x/minfig{seed:03d}.png")
