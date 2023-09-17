import sys, os
import numpy as np
import scipy.special
import scipy.spatial
import scipy.optimize

# must have latex suite installed in system to enable
# matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt

from model import Model
from db import DB

dtype = np.float64


class VDBGLM(DB):
    def __init__(self, T: int, model: Model, seed: int) -> None:
        super().__init__(T, model, seed)
        self.d = model.d
        self.K = model.K
        self.g_z = self.model.g_z
        self.g_z_outer = self.model.g_z_outer
        self.lmbda = 0.001
        self.kappa = 0.1
        self.L_mu = 0.25
        # M_mu = 0.25
        self.delta = 1.0 / self.T
        d = self.d
        K = self.K

        # \sigma_0 is the smallest variance
        self.sigma_0 = 1.0 / (2**6)
        # L is total number of layers
        L = int(np.ceil(np.log2(1.0 / 2 / self.sigma_0)))
        if self.L_override():
            L = 1
        self.L = L
        self.sigmas = self.sigma_0 * np.power(2.0, np.arange(-1, L + 1))
        self.sigmas[0] = 0

        # keey a copy of current estimate
        self.theta = np.ones((L + 1, d))
        self.V = np.zeros((L + 1, d, d), dtype=dtype) + np.eye(d) * self.lmbda
        self.Vinv = (
            1.0 / self.lmbda * np.zeros((L + 1, d, d), dtype=dtype)
            + np.eye(d) / self.lmbda
        )
        self.z = [np.zeros((0, d), dtype=dtype) for _ in range(L + 1)]
        self.r = [np.array([], dtype=dtype) for _ in range(L + 1)]
        # self.sigma_true = np.zeros((T + 1))
        # self.sigma_bar = np.zeros((T + 1))

        self.var = np.ones((L, K, K), dtype=dtype)
        for l in range(L):
            self.var[l] = np.sqrt(
                (self.g_z.reshape(K, K, 1, d) @ self.Vinv[l])
                @ self.g_z.reshape(K, K, d, 1)
            ).reshape(K, K)

        self.D = np.ones((L, K), dtype=dtype)  # feasible set
        self.Psi = np.zeros((L + 1))  # sample count in each layer

        self.count = np.zeros((K, K))

    def update_stats(self, l, z, r):
        self.V[l] += np.outer(z, z)
        self.Vinv[l] = np.linalg.inv(self.V[l])
        self.z[l] = np.vstack([self.z[l], z])
        self.r[l] = np.append(self.r[l], r)
        self.Psi[l] += 1
        self.MLE(l)

    def estimate(self, r, act, eta_scale=0.006):
        t = self.t
        L = self.L
        d = self.d
        K = self.K
        sigma_0 = self.sigma_0
        lmbda = self.lmbda
        kappa = self.kappa
        L_mu = self.L_mu
        delta = self.delta

        x_i, y_i = act
        x = self.model.cA[x_i]
        y = self.model.cA[y_i]
        z = x - y

        self.update_stats(L, z, r)

        p_hat = self.model.mu(self.theta[L] @ z).reshape((1,))
        b_t = (
            np.sqrt(z.reshape(1, d) @ self.Vinv[L] @ z.reshape(d, 1))
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
        # self.sigma_true[t] = np.sqrt(p[x_i, y_i] * (1 - p[x_i, y_i]))
        # self.sigma_bar[t] = bar_sigma_t

        # layer assignment
        l = np.ceil(np.log2(bar_sigma_t / sigma_0)).astype(np.int32)[0]
        l = np.clip(l, 1, L) - 1
        # l = rng.integers(0, L)
        # print((p_hat * (1 - p_hat)), l, b_t, x_i, y_i)
        self.update_stats(l, z, r)

        eta_tl = (
            16
            * (2**l)
            * sigma_0
            / kappa
            * np.sqrt(
                d
                * np.log(1 + self.Psi[l] / d / lmbda)
                * np.log(4 * self.Psi[l] ** 2 / delta)
            )
        )
        +4 / kappa * np.log(4 * (self.Psi[l] ** 2) / delta)
        # TODO: describe the systematic way to choose this scaling value.
        eta_tl *= eta_scale

        if self.eta_tl_override():
            eta_tl = self.eta_tl_override()

        self.var[l] = np.sqrt(
            (self.g_z.reshape(K, K, 1, d) @ self.Vinv[l]) @ self.g_z.reshape(K, K, d, 1)
        ).reshape(K, K)
        # sanity check
        # print(var[l])
        # for i in range(K):
        #     for j in range(K):
        #         assert (var[l][i, j] - np.sqrt(g_z[i, j] @ self.Vinv[l] @ g_z[i, j])) < 1e-6

        ucb = self.g_z @ self.theta[l] + eta_tl * self.var[l]
        # print(ucb, eta_tl)
        cond = ucb.reshape(K, K) >= 0
        self.D[l] = (
            cond.sum(axis=1) == K
        )  # only ucb >= for all opponent arms is selected

    def MLE(self, l: int = 0) -> None:
        theta_0 = self.theta[l]
        func = lambda theta: (
            self.kappa * self.lmbda * theta @ theta
            + (
                np.log(np.exp(self.z[l] @ theta) + 1)
                - self.r[l].reshape(-1, 1) * self.z[l] @ theta
            ).sum(axis=0)
            / self.t
        )
        grad = lambda theta: (
            2 * self.kappa * self.lmbda * theta
            + (
                (self.model.mu(self.z[l] @ theta) - self.r[l]).reshape(-1, 1)
                * self.z[l]
            )
            .sum(axis=0)
            .flatten()
            / self.t
        )

        # gradient descent
        # for _ in range(50):
        #     step = grad(theta_0)
        #     theta_0 -= 0.01 * step
        # self.theta[l] = theta_0

        # optimizer
        res = scipy.optimize.minimize(
            func,
            theta_0,
            jac=grad,
            method="BFGS",
            options={"disp": False, "gtol": 1e-04},
        )
        self.theta[l] = res.x
        # print(theta_0)
        # print(res.x)
        # print('-------------------------------')

    def next_action(self) -> None:
        K = self.K
        Dt = self.D.sum(axis=0) == self.L
        assert np.any(
            Dt
        )  # make sure at least one pair is available in every layer to be sampled

        mask = Dt.reshape(-1, 1) * Dt.reshape(1, -1)
        Lvar = self.var + 1
        # sel = mask * Lvar.min(axis=0)
        x_i, y_i = np.unravel_index(
            np.argmax(mask * Lvar.min(axis=0), axis=None), (K, K)
        )
        return x_i, y_i

    def summarize(self):
        seed = self.model.seed
        np.set_printoptions(precision=6, suppress=True)
        print("--Ground Truth-------")
        print(self.model.theta_star)
        print("--Estimation-------")
        print(self.theta)
        print("Final Estimation Error")
        print(np.linalg.norm(self.theta - self.model.theta_star, axis=1))
        print("Number of sample per layer", self.Psi)
        # print("layer boundaries", sigmas)
        print("acutal max", (self.model.x_star_idx), "last pair", self.next_action())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        ax1.plot(self.model.R)
        # ax1.plot(x, y1, color="blue", label="Sin(x)")
        # ax1.set_xlabel("X")
        # ax1.set_ylabel("Y")
        # ax1.set_title("First Line")
        # ax1.legend()

        if self.L > 3:
            for l in range(1, self.L + 1):
                suffix = f" L={l}"
                x = np.arange(0, len(self.error[l]))
                ax2.plot(x, self.error[l], label=suffix)
        else:
            ax2.plot(self.error)
        # ax2.plot(x, y2, color="red", label="Cos(x)")
        # ax2.set_xlabel("X")
        # ax2.set_ylabel("Y")
        # ax2.set_title("Second Line")
        ax2.set_ylim(0, 1)
        ax2.legend()

        # Adjust spacing between subplots
        plt.tight_layout()
        output_dir = f"data/{self.__class__.__name__}"
        # output_dir = f"data/vdb-s{eta_scale}"
        # output_dir = f"data/vdb-t{theta_scale}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        np.savez(f"{output_dir}/{seed:03d}", r=self.model.R, error=self.error)
        # plt.show()
        plt.savefig(f"{output_dir}/fig{seed:03d}.png")
        plt.close()
        plt.cla()

    def eta_tl_override(self):
        return False

    def L_override(self):
        return False
