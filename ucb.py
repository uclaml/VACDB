import numpy as np
from vdb import LCDB


class MaxInp(LCDB):
    def get_l_t(self):
        return 1

    def beta_scale(self):
        return 2

    def get_beta_tl(self):
        # according to theorectial value in Saha's paper
        # beta_tl = (
        #     1.0
        #     / self.kappa
        #     * np.sqrt(
        #         self.d / 2 * np.log(1 + 2 * self.t / self.d / self.lmbda)
        #         + np.log(1 / self.delta)
        #     )
        # )
        # according to COLSTIM reported value
        if self.beta_scale():
            scale = self.beta_scale()
        beta_tl = np.sqrt(self.d * np.log(self.T)) * scale
        return beta_tl

    def next_action(self) -> None:
        K = self.K
        Dt = self.D[1]

        mask = Dt.reshape(-1, 1) * Dt.reshape(1, -1)
        Lvar = (
            self.enorm[1] + 1
        )  # increase each value by 1, so those out of feasible set will be not be selected in argmax since those are 0
        # sel = mask * Lvar.min(axis=0)
        x_i, y_i = np.unravel_index(np.argmax(mask * Lvar, axis=None), (K, K))
        return x_i, y_i


class MaxFirstUCBNext(MaxInp):
    def next_action(self) -> None:
        Dt = self.D[1]
        u_hat = self.model.cA @ self.theta[1]
        x_i = np.argmax(u_hat * Dt)

        beta_tl = self.get_beta_tl()
        y_i = np.argmax(u_hat + beta_tl * self.enorm[1][x_i])
        # print(x_i, y_i)
        return x_i, y_i


class MaxFirstRndNext(MaxInp):
    def next_action(self) -> None:
        Dt = self.D[1]
        u_hat = self.model.cA @ self.theta[1]
        x_i = np.argmax(u_hat * Dt)

        ii = np.arange(0, self.K, dtype=np.int32)[Dt]
        y_i = self.model.rng.choice(ii)
        return x_i, y_i


class MaxFirstRowMaxNext(MaxInp):
    def next_action(self) -> None:
        Dt = self.D[1]
        u_hat = self.model.cA @ self.theta[1]
        x_i = np.argmax(u_hat * Dt)

        mask = Dt.reshape(-1, 1) * Dt.reshape(1, -1)
        Lvar = self.enorm + 1
        y_i = np.argmax(Lvar.min(axis=0)[x_i] * Dt)
        # sel = mask * Lvar.min(axis=0)
        return x_i, y_i


class MaxPairUCB(MaxInp):
    def next_action(self) -> None:
        Dt = self.D[1]
        u_hat = self.model.cA @ self.theta[1]
        x_plus_y = u_hat + u_hat[:, None]
        beta_tl = self.get_beta_tl()

        x_i, y_i = np.unravel_index(
            np.argmax((x_plus_y + beta_tl * self.enorm[1]), axis=None), (self.K, self.K)
        )
        return x_i, y_i
