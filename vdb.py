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
    seed = 123
print("seed", seed)
rng = np.random.default_rng(seed)
mu = scipy.special.expit
dtype = np.float64

T = 5000
delta = 1.0 / T
d = 4
sigma_0 = 1.0 / (2 ** 5)
L = int(np.ceil(np.log2(1.0 / 2 / sigma_0)))
lmbda = 0.001
kappa = 0.1
L_mu = 0.25
M_mu = 0.25
K = 32

theta_star = rng.random((d,), dtype=dtype)
theta_star = theta_star / np.sqrt(theta_star @ theta_star)
x_orig = rng.integers(0, 2**d, size=(K,), dtype=np.int32)
cA = ((x_orig.reshape(-1, 1) & (2 ** np.arange(d))) != 0) * 2.0 - 1.0
cA += rng.random(cA.shape) - 1
# cA = rng.random((K, d)) - 1


class VALDB:
    def __init__(self) -> None:
        self.theta = np.ones((L + 1, d))
        self.V = np.zeros((L + 1, d, d), dtype=dtype) + np.eye(d) * lmbda
        self.Vinv = (
            1.0 / lmbda * np.zeros((L + 1, d, d), dtype=dtype) + np.eye(d) / lmbda
        )
        self.xy_diff = [np.zeros((0, d), dtype=dtype) for _ in range(L + 1)]
        self.r = [np.array([], dtype=dtype) for _ in range(L + 1)]
        self.R = np.zeros((T + 1), dtype=dtype)

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
            theta_0 -= 0.0001 * grad(theta_0)
        self.theta[l] = theta_0
        # print(theta_0)

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
            b_t *= 0.01
            # bar_sigma_t = np.sqrt(p_hat * (1 - p_hat) + b_t + b_t**2)
            bar_sigma_t = np.sqrt(p_hat * (1 - p_hat) + 2 * b_t)

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
            eta_tl *= 0.005
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


v = VALDB()
Psi = v.main()
np.set_printoptions(precision=6, suppress=True)
print(v.theta)
print("----------")
print(theta_star)
print("layer sample", Psi)
print(np.linalg.norm(v.theta - theta_star, axis=1))

plt.plot(v.R[1:])
np.savez(f"data/R{seed:03d}")
# plt.show()
plt.savefig(f"data/fig{seed:03d}.png")
