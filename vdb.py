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
    seed = 60
print("seed", seed)
rng = np.random.default_rng(seed)
mu = scipy.special.expit
dtype = np.float64

# TODO: 50000
T = 2500
delta = 1.0 / T
d = 10
sigma_0 = 1.0 / (2**6)
L = int(np.ceil(np.log2(1.0 / 2 / sigma_0)))
sigmas = sigma_0 * np.power(2., np.arange(-1, L + 1))
sigmas[0] = 0
lmbda = 1
kappa = 0.1
L_mu = 0.25
M_mu = 0.25
K = 50

theta_star = rng.random((d,), dtype=dtype)
theta_star = theta_star / np.sqrt(theta_star @ theta_star)
theta_star *= 3
x_orig = rng.integers(0, 2**d, size=(K,), dtype=np.int32)
# cA[0:K//4] += rng.random((K//4, d)) * 2
cA = ((x_orig.reshape(-1, 1) & (2 ** np.arange(d))) != 0) * 2.0 - 1.0
cA += rng.random(cA.shape) - 1
# cA = rng.random((K, d)) - 1


class VDBGLM:
    def __init__(self) -> None:
        self.theta = np.ones((L + 1, d))
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
            2 * kappa * lmbda * theta
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
        self.g_xy_diff  = g_xy_diff
        p = mu(g_xy_diff @ theta_star)
        # sanity check
        # for i in range(K):
        #     for j in range(K):
        #         assert np.all(g_xy_diff[i, j] == cA[i] - cA[j])
        #         assert np.all(
        #             g_xy_diff_outer[i, j] == np.outer(cA[i] - cA[j], cA[i] - cA[j])
        #         )
        #         assert (p[i, j] - mu(cA[i] @ theta_star - cA[j] @ theta_star)) < 1e-10
        print(p)
        print(cA)
        print(theta_star)

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
            Lvar = var + 1
            sel = mask * Lvar.min(axis=0)
            x_i, y_i = np.unravel_index(
                np.argmax(mask * Lvar.min(axis=0), axis=None), (K, K)
            )
            if (x_i, y_i) == (2, 14):
                pass
            # x_i = rng.integers(0, K)
            # y_i = rng.integers(0, K)
            x = cA[x_i]
            y = cA[y_i]
            xy_diff = x - y
            r = rng.binomial(1, p[x_i, y_i])
            self.R[t] = (2 * cA[x_star_idx] - x - y) @ theta_star + self.R[t - 1]
            self.V[L] += g_xy_diff_outer[x_i, y_i]
            self.Vinv[L] = np.linalg.inv(self.V[L])
            self.xy_diff[L] = np.vstack([self.xy_diff[L], xy_diff])
            self.r[L] = np.append(self.r[L], r)
            cPsi[L] += 1
            self.MLE(L)

            p_hat = mu(self.theta[L] @ xy_diff).reshape((1,))
            b_t = (
                np.sqrt(xy_diff.reshape(1, d) @ self.Vinv[L] @ xy_diff.reshape(d, 1))
                * L_mu
                * (
                    1.0 / kappa * np.sqrt(d * np.log(1 / delta + t / lmbda / delta))
                    + np.sqrt(lmbda)
                )
            ).reshape((1,))
            # check the bar sigma v.s. sigma
            # remark: if the estimation is wrong, the only bad thing happens is the sample is put to a wrong layer
            # which just increase the sample commplexity
            b_t *= 0.01
            # bar_sigma_t = np.sqrt(p_hat * (1 - p_hat) + b_t + b_t**2)
            bar_sigma_t = np.sqrt(p_hat * (1 - p_hat) + 2 * b_t)
            self.sigma_true[t] = np.sqrt(p[x_i, y_i] * (1 - p[x_i, y_i]))
            self.sigma_bar[t] = bar_sigma_t

            l = np.ceil(np.log2(bar_sigma_t / sigma_0)).astype(np.int32)[0]
            l = np.clip(l, 1, L) - 1
            # l = rng.integers(0, L)
            # print((p_hat * (1 - p_hat)), l, b_t, x_i, y_i)
            self.V[l] += g_xy_diff_outer[x_i, y_i]
            self.Vinv[l] = np.linalg.inv(self.V[l])
            self.xy_diff[l] = np.vstack([self.xy_diff[l], xy_diff])
            self.r[l] = np.append(self.r[l], r)
            cPsi[l] += 1
            self.MLE(l)

            # check the other eta_tl calc
            eta_tl = (
                16
                * (2**l)
                * sigma_0
                / kappa
                * np.sqrt(
                    d
                    * np.log(1 + cPsi[l] / d / lmbda)
                    * np.log(4 * cPsi[l] ** 2 / delta)
                )
            )
            +4 / kappa * np.log(4 * (cPsi[l] ** 2) / delta)
            # calculation in Saha's paper
            # eta_tl = (
            #     1.0 / kappa * np.sqrt(d / 2 * np.log(1 + 2 * t / d / lmbda) + np.log(1 / delta))
            # )
            # TODO: describe the systematic way to choose this scaling value.
            eta_tl *= .008
            var[l] = np.sqrt(
                (g_xy_diff.reshape(K, K, 1, d) @ self.Vinv[l])
                @ g_xy_diff.reshape(K, K, d, 1)
            ).reshape(K, K)
            # sanity check
            # print(var[l])
            # for i in range(K):
            #     for j in range(K):
            #         assert (var[l][i, j] - np.sqrt(g_xy_diff[i, j] @ self.Vinv[l] @ g_xy_diff[i, j])) < 1e-6
            ucb = g_xy_diff @ self.theta[l] + eta_tl * var[l]
            # print(ucb, eta_tl)
            cond = ucb.reshape(K, K) >= 0
            D[l] = cond.sum(axis=1) == K

            print(f"{t}", end="\r")

        print(x_star_idx, "x_star")
        return cPsi


v = VDBGLM()
Psi = v.main()
# v.MLE(0)
# v.MLE(1)
np.set_printoptions(precision=6, suppress=True)
print(v.theta)
print("----------")
print(theta_star)
print("layer sample", Psi)
print("layer boundaries", sigmas)
print(np.linalg.norm(v.theta - theta_star, axis=1))

# plt.plot(v.sigma_bar[1:] - v.sigma_true[1:])
# plt.show()
# plt.savefig("sigma-bar-true-005.png")

plt.plot(v.R[1:])
np.savez(f"data/vdb-3x/R{seed:03d}", r=v.R)
# plt.show()
plt.savefig(f"data/vdb-3x/fig{seed:03d}.png")
plt.close()
plt.cla()
p = mu(v.g_xy_diff @ theta_star).flatten()
sigma_true = np.sqrt(p * (1 - p))
plt.hist(sigma_true, bins=sigmas, edgecolor='black')
plt.savefig(f"data/vdb-3x/his{seed:03d}.png")
