import numpy as np
from vdb import LCDB
from model import DTYPE, Model

import scipy.special


class AdaCDB(LCDB):
    def init_sigma(self):
        self.beta_hist = [[] for _ in range(self.L + 1)]

        d = self.d
        K = self.K
        L = self.L
        self.lmbda = 0.1
        self.kappa = 0.01
        self.L_mu = 1

        self.xpy = self.model.cA.reshape(K, 1, d) + self.model.cA.reshape(1, K, d)
        # self.Sigma = np.zeros((L + 1, d, d), dtype=DTYPE) + np.eye(d) * self.lmbda * np.power(2.0, -2 * np.arange(0, L + 1)).reshape(L + 1, 1, 1)
        # self.SigmaInv = 1.0 / self.lmbda * np.zeros((L + 1, d, d), dtype=DTYPE) + np.eye(d) / self.lmbda * np.power(2.0, 2 * np.arange(0, L + 1)).reshape(L + 1, 1, 1)
        self.Sigma = np.zeros((L + 1, d, d), dtype=DTYPE) + np.eye(d) * np.power(
            2.0, -2 * np.arange(0, L + 1)
        ).reshape(L + 1, 1, 1)
        self.SigmaInv = 1.0 * np.zeros((L + 1, d, d), dtype=DTYPE) + np.eye(
            d
        ) * np.power(2.0, 2 * np.arange(0, L + 1)).reshape(L + 1, 1, 1)

        self.beta_t = 0.05 * np.power(2.0, -np.arange(0, L + 1) + 1)

    def get_l_t(self, act) -> (bool, int):
        # AdaCDB
        l = 1
        stop = False
        while l <= self.L:
            if self.enorm[l, act[0], act[1]] >= np.power(2.0, -l):
                break
            else:
                l += 1
        if l == self.L + 1:
            stop = True
        return stop, l

    def get_beta_tl(self):
        return np.sqrt(self.d * np.log(self.T)) * 2 / 8

    def get_beta_tl_theory(self, l):
        log_t1Ldl = np.log(4 * (self.t + 1) ** 2 * self.L / self.delta)
        log_tLdl = np.log(4 * (self.t) ** 2 * self.L / self.delta)
        # log_t1Ldl = 1
        # log_tLdl = 1
        Var_tl = self.Psi[l]
        if np.power(2.0, l) >= 64 * self.L_mu / self.kappa * np.sqrt(log_t1Ldl):
            Var_tl = (
                np.power(
                    self.w[l]
                    * self.w[l]
                    * (self.model.mu(self.z[l] @ self.theta) - self.r[l]),
                    2,
                ).reshape(-1, 1)
            ).sum(axis=0)
        inner = (Var_tl * 8 + 18 * log_t1Ldl) * log_tLdl
        beta_tl = (
            16 * np.power(2.0, -l) * np.sqrt(inner) / self.kappa / 10000
            + 6 * np.power(2.0, -l) / self.kappa * log_tLdl / 10000
            + np.power(2.0, -l + 1)
        )
        # print(Var_tl, np.sqrt(inner), log_t1Ldl, log_tLdl, self.beta_t[l])
        return beta_tl

    def update_stats(self, l, z, r, w):
        K = self.K
        d = self.d
        # print(l, self.beta_t[l] * self.enorm[l, x_i, y_i], w, act)
        # print(self.xpy[x_i, y_i] @ self.theta[l], self.model.x_star_idx)
        self.Sigma[l] += np.outer(z, z) * w * w
        self.SigmaInv[l] = np.linalg.inv(self.Sigma[l])
        self.enorm[l] = (
            np.sqrt(
                self.g_z.reshape(K, K, 1, d)
                @ self.SigmaInv[l]
                @ self.g_z.reshape(K, K, d, 1)
            )
        ).reshape(K, K)

        self.z[l] = np.vstack([self.z[l], z])
        self.r[l] = np.append(self.r[l], r)
        self.w[l] = np.append(self.w[l], w)
        self.Psi[l] += 1
        self.MLE(l)

    def estimate(self, r, act):
        stop, l = self.get_l_t(act)
        if stop:
            return

        x_i, y_i = act
        z = self.model.cA[x_i] - self.model.cA[y_i]
        w = np.power(2.0, -l) / self.enorm[l, x_i, y_i]

        self.update_stats(l, z, r, w)
        self.update_stats(0, z, r, w)

        # self.beta_t[l] = self.get_beta_tl_theory(l) / self.model.scale
        self.beta_t[l] = self.get_beta_tl() / self.model.scale

        for ll in range(1, self.L + 1):
            self.beta_hist[ll].append(0)
        self.beta_hist[l][-1] = self.beta_t[l]

        self.count_xyL[l, x_i, y_i] += 1
        self.count_xy[x_i, y_i] += 1

    def MLE(self, l: int = 0) -> None:
        theta_0 = self.theta[l]
        # use last layer's estimate instead of starting from scrach
        if (l >= 2) and (np.linalg.norm(self.theta[l]) == 0):
            theta_0 = self.theta[l - 1]
        func = lambda theta: (
            self.kappa * np.power(2.0, -2 * l) * theta @ theta
            + (
                self.w[l]
                * self.w[l]
                * (
                    np.log(np.exp(self.z[l] @ theta) + 1)
                    - self.r[l].reshape(-1, 1) * self.z[l] @ theta
                )
            ).sum(axis=0)
            # / self.t
        )
        # T T T,d@d, -> T, -  T,1*T,d->T,d * T,d @d -> T
        grad = lambda theta: (
            2 * self.kappa * np.power(2.0, -2 * l) * theta
            + (
                (
                    self.w[l]
                    * self.w[l]
                    * (self.model.mu(self.z[l] @ theta) - self.r[l])
                ).reshape(-1, 1)
                * self.z[l]
            )
            .sum(axis=0)
            .flatten()
            # / self.t
        )
        # print("I want to see the grad:", grad(theta_0))

        # gradient descent
        # for _ in range(50):
        #     step = grad(theta_0)
        #     theta_0 -= 0.01 * step
        # self.theta[l] = theta_0

        # second order optimizer
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
        b = self.beta_t.reshape(L + 1, 1, 1) * self.enorm  # L,1,1 * L,K,K
        value = np.min(a + b, axis=0)
        x_i, y_i = np.unravel_index(np.argmax(value, axis=None), (K, K))
        # print(x_i, y_i)
        return x_i, y_i


class VACDB(AdaCDB):
    def __init__(self, T: int, model: Model, L: int) -> None:
        super().__init__(T, model, L)
        self.l = 1

    def get_l_t(self, _) -> (bool, int):
        # VACDB
        # TODO: pass l as argument?
        # reached the top most layer
        if self.l > self.L:
            # print("here")
            stop = True
            l = self.L
        else:
            l = self.l
            stop = False
        return stop, l

    def next_action(self):
        K = self.K
        d = self.d
        L = self.L

        l = 1
        Dt = [np.ones(K, dtype=np.int64) for _ in range(L + 1)]
        while True:
            mask = Dt[l].reshape(K, 1) * Dt[l].reshape(1, K)
            # print("arm remaining", Dt[l].sum(axis=0))
            # print("best arm still here", Dt[l][12])
            if np.all(mask * self.enorm[l] <= self.alpha):
                u_hat = self.model.cA @ self.theta[l]
                x_plus_y = u_hat + u_hat[:, None]
                cb = self.enorm[l]
                beta_tl = self.beta_t[l]
                cond = x_plus_y + beta_tl * cb
                cond[mask < 1] = -11111111
                x_i, y_i = np.unravel_index(np.argmax(cond, axis=None), (K, K))
                self.l = l + 1
                # print("best", x_i, y_i)
                break
            elif np.all(mask * self.enorm[l] <= np.power(2.0, -l)):
                u_hat = self.model.cA @ self.theta[l]
                u_hat_max = np.max(u_hat[Dt[l] > 0])
                cond = u_hat - u_hat_max + np.power(2.0, -l) * self.beta_t[l] >= 0
                if l + 1 <= L:
                    Dt[l + 1] = cond * Dt[l]
                l = l + 1
            else:
                sel_mat = mask * self.enorm[l] > np.power(2.0, -l)
                # depth first explore
                # x_i, y_i = np.unravel_index(np.argmax(sel_mat, axis=None), (K, K))
                # uniform explore: breadth first
                choices = np.arange(K * K).reshape(K, K)[sel_mat].flatten()
                choice = self.model.rng.choice(choices)
                x_i, y_i = np.unravel_index(choice, (K, K))
                self.l = l
                break

        return x_i, y_i


class StaDSim(VACDB):
    def get_beta_tl(self):
        return super().get_beta_tl() * 8

    def estimate(self, r, act):
        stop, l = self.get_l_t(act)
        if stop:
            return

        x_i, y_i = act
        z = self.model.cA[x_i] - self.model.cA[y_i]
        w = 1

        self.update_stats(l, z, r, w)
        self.update_stats(0, z, r, w)

        # self.beta_t[l] = self.get_beta_tl_theory(l) / self.model.scale
        self.beta_t[l] = self.get_beta_tl() / self.model.scale

        for ll in range(1, self.L + 1):
            self.beta_hist[ll].append(0)
        self.beta_hist[l][-1] = self.beta_t[l]

        self.count_xyL[l, x_i, y_i] += 1
        self.count_xy[x_i, y_i] += 1


class StaD(StaDSim):
    def __init__(self, T: int, model: Model, L: int) -> None:
        L = 10
        super().__init__(T, model, L)
        self.alpha = 1.0 / np.sqrt(self.T)

    # def get_beta_tl(self):
    #     return np.sqrt(self.d * np.log(self.T)) * 2

    def next_action(self):
        K = self.K
        d = self.d
        L = self.L

        l = 1
        # \cG_t
        Dt = [np.ones(K, dtype=np.int64) for _ in range(L + 1)]
        while True:
            mask = Dt[l].reshape(K, 1) * Dt[l].reshape(1, K)
            # print("arm remaining", Dt[l].sum(axis=0))
            # print("best arm still here", Dt[l][12])
            if np.all(mask * self.enorm[l] <= self.alpha / self.beta_t[l]):
                u_hat = self.model.cA @ self.theta[l]
                x_i = np.argmax(u_hat * Dt[l])
                y_i = np.argmax((u_hat + self.beta_t[l] * self.enorm[l][x_i]) * Dt[l])

                self.l = l + 1
                # print("best", x_i, y_i)
                break
            elif np.all(mask * self.enorm[l] <= np.power(2.0, -l)):
                u_hat = self.model.cA @ self.theta[l]
                u_hat_max = np.max(u_hat[Dt[l] > 0])
                cond = u_hat - u_hat_max + np.power(2.0, -l) * self.beta_t[l] >= 0
                if l + 1 <= L:
                    Dt[l + 1] = cond * Dt[l]
                l = l + 1
            else:
                sel_mat = mask * self.enorm[l] > np.power(2.0, -l)
                # depth first explore
                # x_i, y_i = np.unravel_index(np.argmax(sel_mat, axis=None), (K, K))
                # uniform explore: breadth first
                choices = np.arange(K * K).reshape(K, K)[sel_mat].flatten()
                choice = self.model.rng.choice(choices)
                x_i, y_i = np.unravel_index(choice, (K, K))
                self.l = l
                break

        return x_i, y_i
