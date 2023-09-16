import numpy as np
from db import DB
from model import Model

import scipy.special
import matplotlib.pyplot as plt

import os, sys
dtype = np.float64


class AdaDBGLM(DB):
    def __init__(self, T: int, model: Model, seed: int, L: int = 6) -> None:
        super().__init__(T, model, seed)

        self.L = L
        self.d = model.d
        self.K = model.K
        self.g_z = self.model.g_z
        self.g_z_outer = self.model.g_z_outer
        self.kappa = 0.1

        d = self.d
        K = self.K
        self.lmbda = 0.1
        self.xpy = self.model.cA.reshape(K, 1, d) + self.model.cA.reshape(1, K, d)
        self.Sigma = np.zeros((L + 1, d, d), dtype=dtype) + np.eye(
            d
        ) * self.lmbda * 1.111 * np.power(2.0, -2 * np.arange(0, L + 1)).reshape(
            L + 1, 1, 1
        )
        self.SigmaInv = 1.0 / self.lmbda * np.zeros(
            (L + 1, d, d), dtype=dtype
        ) + np.eye(d) / 1.111 / self.lmbda * np.power(
            2.0, 2 * np.arange(0, L + 1)
        ).reshape(
            L + 1, 1, 1
        )
        self.z = [np.zeros((0, d), dtype=dtype) for _ in range(L + 1)]
        self.r = [np.array([], dtype=dtype) for _ in range(L + 1)]
        self.w = [np.array([], dtype=dtype) for _ in range(L + 1)]

        self.var = np.ones((L + 1, K, K), dtype=dtype)
        for l in range(L + 1):
            self.var[l] = np.sqrt(
                (self.g_z.reshape(K, K, 1, d) @ self.SigmaInv[l])
                @ self.g_z.reshape(K, K, d, 1)
            ).reshape(K, K)

        self.theta = np.zeros((L + 1, d)) / d
        self.beta = 1 * np.power(2.0, -np.arange(0, L + 1) + 1)

        # for l in range(L + 1):
        #     tmp = self.beta[l] * \
        #     np.sqrt(np.ones((1, d)) @ self.SigmaInv[l] @ np.ones((d, 1)))
        #     print(tmp)
        # exit()

    def estimate(self, r, act):
        K = self.K
        d = self.d
        l = 1
        while l <= self.L:
            if self.var[l, act[0], act[1]] >= np.power(2.0, -l):
                break
            else:
                l += 1
        if l == self.L + 1:
            return

        x_i, y_i = act
        z = self.model.cA[x_i] - self.model.cA[y_i]
        w = np.power(2.0, -l) / self.var[l, x_i, y_i]
        print(l, self.beta[l] * self.var[l, x_i, y_i], w, act)
        print(self.xpy[x_i, y_i] @ self.theta[l], self.model.x_star_idx)
        self.Sigma[l] += np.outer(z, z) * w * w
        self.SigmaInv[l] = np.linalg.inv(self.Sigma[l])
        self.var[l] = (
            np.sqrt(
                self.g_z.reshape(K, K, 1, d)
                @ self.SigmaInv[l]
                @ self.g_z.reshape(K, K, d, 1)
            )
        ).reshape(K, K)

        self.z[l] = np.vstack([self.z[l], z])
        self.r[l] = np.append(self.r[l], r)
        self.w[l] = np.append(self.w[l], w)
        # self.Psi[l] += 1
        self.MLE(l)
        self.beta[l] = 1 * np.power(2.0, -l) * np.sqrt(np.log(self.T))

    def MLE(self, l: int = 0) -> None:
        theta_0 = self.theta[l]
        if((l > 1) and (np.linalg.norm(self.theta[l]) == 0)):
            theta_0 = self.theta[l-1]
        func = lambda theta: (
            self.kappa * np.power(2.0, -2 * l) * theta @ theta
            + ( self.w[l] * self.w[l] *(
                np.log(np.exp(self.z[l] @ theta) + 1)
                # (self.z[l] @ theta)
                - self.r[l].reshape(-1, 1) * self.z[l] @ theta
                )
            ).sum(axis=0)
            #/ self.t
        )
        # T T T,d@d, -> T, -  T,1*T,d->T,d * T,d @d -> T
        grad = lambda theta: (
            2 * self.kappa * np.power(2.0, -2*l) * theta
            + (
                (
                    self.w[l] * self.w[l] *(
                    self.model.mu(self.z[l] @ theta)
                    - self.r[l]
                    )
                ).reshape(-1, 1)
                * self.z[l]
            )
            .sum(axis=0)
            .flatten()
            #/ self.t
        )
        #print("I want to see the grad:", grad(theta_0))

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

    def next_action(self):
        K = self.K
        d = self.d
        L = self.L
        a = np.zeros((L + 1, K, K))
        for l in range(L):
            a[l] = (self.xpy @ self.theta[l].T).reshape(
                K, K
            )  # 1,K,K,d @ L,1,1,d = L,K,K
        b = self.beta.reshape(L + 1, 1, 1) * self.var  # L,1,1 * L,K,K
        view = self.beta[l] * self.var[l, 1, 1]
        # print(view)
        value = np.min(a + b, axis=0)
        x_i, y_i = np.unravel_index(np.argmax(value, axis=None), (K, K))
        # print(x_i, y_i)
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
        # print("Number of sample per layer", self.Psi)
        # print("layer boundaries", sigmas)
        print("acutal max", (self.model.x_star_idx))
        output_dir = f"data/{self.__class__.__name__}"
        # output_dir = f"data/vdb-s{eta_scale}"
        # output_dir = f"data/vdb-t{theta_scale}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        np.savez(
            f"{output_dir}/{seed:03d}",
            r=self.model.R,
            error=self.model.error,
            Lerr=self.model.Lerr,
        )
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        ax1.plot(self.model.R)
        # ax1.plot(x, y1, color="blue", label="Sin(x)")
        # ax1.set_xlabel("X")
        # ax1.set_ylabel("Y")
        # ax1.set_title("First Line")
        # ax1.legend()

        ax2.plot(self.model.error)
        ax2.set_ylim(0, 1)
        # ax2.plot(x, y2, color="red", label="Cos(x)")
        # ax2.set_xlabel("X")
        # ax2.set_ylabel("Y")
        # ax2.set_title("Second Line")
        # ax2.legend()

        # Adjust spacing between subplots
        plt.tight_layout()
        output_dir = f"data/{self.__class__.__name__}"
        # output_dir = f"data/vdb-s{eta_scale}"
        # output_dir = f"data/vdb-t{theta_scale}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        np.savez(f"{output_dir}/{seed:03d}", r=self.model.R, error=self.model.error)
        # plt.show()
        plt.savefig(f"{output_dir}/fig{seed:03d}.png")
        plt.close()
        plt.cla()


class SAVE(AdaDBGLM):
    def __init__(self, T: int, model: Model, seed: int, L: int = 7) -> None:
        super().__init__(T, model, seed)

        self.L = L
        self.d = model.d
        self.K = model.K
        self.g_z = self.model.g_z
        self.g_z_outer = self.model.g_z_outer
        self.kappa = 0.1
        self.alpha = np.power(2.0, -L)

        d = self.d
        K = self.K
        self.lmbda = 0.1
        self.xpy = self.model.cA.reshape(K, 1, d) + self.model.cA.reshape(1, K, d)
        self.Sigma = np.zeros((L + 1, d, d), dtype=dtype) + np.eye(d) * np.power(
            2.0, -2 * np.arange(0, L + 1)
        ).reshape(L + 1, 1, 1)
        self.SigmaInv = 1.0 * np.zeros((L + 1, d, d), dtype=dtype) + np.eye(
            d
        ) * np.power(2.0, 2 * np.arange(0, L + 1)).reshape(L + 1, 1, 1)
        self.z = [np.zeros((0, d), dtype=dtype) for _ in range(L + 1)]
        self.r = [np.array([], dtype=dtype) for _ in range(L + 1)]
        self.w = [np.array([], dtype=dtype) for _ in range(L + 1)]
        self.Phi = np.zeros(L + 1)

        self.var = np.ones((L + 1, K, K), dtype=dtype)
        for l in range(L + 1):
            self.var[l] = np.sqrt(
                (self.g_z.reshape(K, K, 1, d) @ self.SigmaInv[l])
                @ self.g_z.reshape(K, K, d, 1)
            ).reshape(K, K)

        self.theta = np.zeros((L + 1, d)) / d
        self.beta = 0.1 * np.power(2.0, -np.arange(0, L + 1) + 1)

    def estimate(self, r, act):
        K = self.K
        d = self.d
        # TODO: pass l as argument?
        l = self.l
        if not l:
            # change beta?
            return
        x_i, y_i = act
        z = self.model.cA[x_i] - self.model.cA[y_i]
        w = 1 * np.power(2.0, -l) / self.var[l, x_i, y_i]
        #w = 1
        print(l, self.beta[l] * self.var[l, x_i, y_i], w, act)
        print(self.xpy[x_i, y_i] @ self.theta[l], self.model.x_star_idx)
        self.Sigma[l] += np.outer(z, z) * w * w * (3.2 ** l) * 0.5
        self.SigmaInv[l] = np.linalg.inv(self.Sigma[l])
        self.var[l] = (
            np.sqrt(
                self.g_z.reshape(K, K, 1, d)
                @ self.SigmaInv[l]
                @ self.g_z.reshape(K, K, d, 1)
            )
        ).reshape(K, K)

        self.z[l] = np.vstack([self.z[l], z])
        self.r[l] = np.append(self.r[l], r)
        self.w[l] = np.append(self.w[l], w)
        # self.Psi[l] += 1
        self.MLE(l)
        self.beta[l] = 0.05 * np.power(2.0, -l) * np.sqrt(np.log(self.t))

    def next_action(self):
        K = self.K
        d = self.d
        L = self.L

        l = 1
        Dt = [np.ones(K, dtype=np.int64) for _ in range(L + 1)]
        while True:
            mask = Dt[l].reshape(K, 1) * Dt[l].reshape(1, K)
            print("arm remaining", Dt[l].sum(axis=0))
            print("best arm still here", Dt[l][12])
            if np.all(mask * self.var[l] <= self.alpha):
                u_hat = self.model.cA @ self.theta[l]
                x_plus_y = u_hat + u_hat[:, None]
                cb = self.var[l]
                eta_tl = self.beta[l]
                cond = x_plus_y + eta_tl * cb
                cond[~mask] = -11111111
                x_i, y_i = np.unravel_index(np.argmax(cond, axis=None), (K, K))
                self.l = None
                print("best", x_i, y_i)
                break
            elif np.all(mask * self.var[l] <= np.power(2.0, -l)):
                u_hat = self.model.cA @ self.theta[l]
                u_hat_max = np.max(u_hat[Dt[l]])
                cond = u_hat - u_hat_max + np.power(2.0, -l) * self.beta[l] >= 0
                if l + 1 <= L:
                    Dt[l + 1] = cond * Dt[l]
            else:
                sel_mat = mask * self.var[l] 
                #> np.power(2.0, -l)
                x_i, y_i = np.unravel_index(np.argmax(sel_mat, axis=None), (K, K))
                # choices = np.arange(K * K).reshape(K, K)[sel_mat].flatten()
                # choice = self.rng.choice(choices)
                # x_i, y_i = np.unravel_index(
                #     choice, (K, K)
                # )
                self.l = l
                break
            l = l + 1

        return x_i, y_i
