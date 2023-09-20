import sys, os
import numpy as np
import scipy.special
import scipy.spatial
import scipy.optimize

# must have latex suite installed in system to enable
# matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt

from model import Model, DTYPE
from db import DB


class LCDB(DB):
    """variance aware contextual bandits"""

    def init_sigma(self):
        L = self.L
        d = self.d

        self.lmbda = 0.001
        self.kappa = 0.1
        self.L_mu = 0.25
        # self.M_mu = 0.25

        self.Sigma = np.zeros((L + 1, d, d), dtype=DTYPE) + np.eye(d) * self.lmbda
        self.SigmaInv = (
            1.0 / self.lmbda * np.zeros((L + 1, d, d), dtype=DTYPE)
            + np.eye(d) / self.lmbda
        )
        # \sigma_0(alpha) is the smallest variance
        self.sigmas = self.alpha * np.power(2.0, np.arange(-1, L + 1))
        self.sigmas[0] = 0

    def update_stats(self, l, z, r):
        self.Sigma[l] += np.outer(z, z)
        self.SigmaInv[l] = np.linalg.inv(self.Sigma[l])
        self.z[l] = np.vstack([self.z[l], z])
        self.r[l] = np.append(self.r[l], r)
        self.Psi[l] += 1
        self.MLE(l)

    def estimate(self, r, act, beta_scale=0.006):
        t = self.t
        L = self.L
        d = self.d
        K = self.K
        sigma_0 = self.alpha
        lmbda = self.lmbda
        kappa = self.kappa
        L_mu = self.L_mu
        delta = self.delta

        x_i, y_i = act
        x = self.model.cA[x_i]
        y = self.model.cA[y_i]
        z = x - y

        # global estimator
        self.update_stats(0, z, r)
        self.count_xy[x_i, y_i] += 1

        p_hat = self.model.mu(self.theta[L] @ z).reshape((1,))
        b_t = (
            np.sqrt(z.reshape(1, d) @ self.SigmaInv[L] @ z.reshape(d, 1))
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
        l = np.clip(l, 1, L)
        # l = self.model.rng.integers(1, L + 1)
        # print((p_hat * (1 - p_hat)), l, b_t, x_i, y_i)
        self.update_stats(l, z, r)
        self.count_xyL[l, x_i, y_i] += 1

        beta_tl = (
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
        beta_tl *= beta_scale

        if self.beta_tl_override():
            beta_tl = self.beta_tl_override()

        self.enorm[l] = np.sqrt(
            (self.g_z.reshape(K, K, 1, d) @ self.SigmaInv[l])
            @ self.g_z.reshape(K, K, d, 1)
        ).reshape(K, K)
        # sanity check
        # print(var[l])
        # for i in range(K):
        #     for j in range(K):
        #         assert (var[l][i, j] - np.sqrt(g_z[i, j] @ self.SigmaInv[l] @ g_z[i, j])) < 1e-6

        ucb = self.g_z @ self.theta[l] + beta_tl * self.enorm[l]
        # print(ucb, beta_tl)
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
        Dt = self.D[1:].sum(axis=0) == self.L
        assert np.any(
            Dt
        )  # make sure at least one pair is available in every layer to be sampled

        mask = Dt.reshape(-1, 1) * Dt.reshape(1, -1)
        Lvar = self.enorm + 1
        # sel = mask * Lvar.min(axis=0)
        x_i, y_i = np.unravel_index(
            np.argmax(mask * Lvar.min(axis=0), axis=None), (K, K)
        )
        return x_i, y_i

    def summarize(self, suffix=""):
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

        fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        (fig1, fig2) = fig.subfigures(2, 1, wspace=0.07, height_ratios=[2, 1])

        fig1subs = fig1.subplots(1, 2)
        ax1, ax2 = fig1subs
        ax1.plot(self.model.R)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Regret(t)")

        if self.L >= 2:
            for l in range(1, self.L + 1):
                legend_suffix = f" L={l}"
                x = np.arange(0, len(self.error[l]))
                ax2.plot(x, self.error[l], label=legend_suffix)
        else:
            print(len(self.error))
            print(len(self.error[0]))
            x = np.arange(0, len(self.error[1]))
            ax2.plot(x, self.error[1], label="Single Layer")
        ax2.plot(x, np.abs(np.array(self.error[0]) - 0.5), label="|p_xy - 0.5|")
        ax2.set_ylim(0, 1)
        ax2.legend()

        axes = fig2.subplots(1, 5)
        axes[0].matshow(np.log((1 + self.count_xy)))
        axes[0].set_title("Global counts")
        for l in range(1, 5):
            if l <= self.L:
                axes[l].matshow(np.log((1 + self.count_xyL[l])))
            else:
                axes[l].matshow(np.zeros((self.K, self.K)))
            axes[l].set_title(f"Layer {l} counts")

        plt.tight_layout()
        output_dir = f"data/{self.__class__.__name__}" + suffix
        # output_dir = f"data/vdb-s{beta_scale}"
        # output_dir = f"data/vdb-t{thbeta_scale}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        np.savez(f"{output_dir}/{seed:03d}", r=self.model.R, error=self.error)
        # plt.show()
        plt.savefig(f"{output_dir}/fig{seed:03d}.png")
        plt.close()
        plt.cla()

    def beta_tl_override(self):
        return False

    def single_layer(self):
        return False
