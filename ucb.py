import numpy as np
from vdb import LCDB


class MaxInp(LCDB):
    def beta_scale(self):
        return 2

    def beta_tl_override(self):
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

    def single_layer(self):
        return True

    def next_action(self) -> None:
        return super().next_action()


class MaxFirstUCBNext(MaxInp):
    def next_action(self) -> None:
        Dt = self.D[1:].sum(axis=0) == self.L
        assert np.any(
            Dt
        )  # make sure at least one pair is available in every layer to be sampled

        u_hat = self.model.cA @ self.theta[-1]
        x_i = np.argmax(u_hat * Dt)

        mask = Dt.reshape(-1, 1) * Dt.reshape(1, -1)
        Lvar = self.enorm
        cb = Lvar.min(axis=0)
        beta_tl = self.beta_tl_override()
        y_i = np.argmax(u_hat + beta_tl * cb[x_i])
        # print(x_i, y_i)
        return x_i, y_i


class MaxFirstRndNext(MaxInp):
    def next_action(self) -> None:
        Dt = self.D[1:].sum(axis=0) == self.L
        assert np.any(
            Dt
        )  # make sure at least one pair is available in every layer to be sampled

        u_hat = self.model.cA @ self.theta[-1]
        x_i = np.argmax(u_hat * Dt)
        ii = np.arange(0, self.K, dtype=np.int32)[Dt]
        y_i = self.model.rng.choice(ii)
        # print(x_i, y_i)
        return x_i, y_i


class MaxFirstRowMaxNext(MaxInp):
    def next_action(self) -> None:
        Dt = self.D.sum(axis=0) == self.L
        assert np.any(
            Dt
        )  # make sure at least one pair is available in every layer to be sampled

        u_hat = self.model.cA @ self.theta[-1]
        x_i = np.argmax(u_hat * Dt)
        mask = Dt.reshape(-1, 1) * Dt.reshape(1, -1)
        Lvar = self.enorm + 1
        y_i = np.argmax(Lvar.min(axis=0)[x_i] * Dt)
        # sel = mask * Lvar.min(axis=0)
        return x_i, y_i


class MaxPairUCB(MaxInp):
    def single_layer(self):
        return True

    def next_action(self) -> None:
        K = self.K
        Dt = self.D[1:].sum(axis=0) == self.L
        assert np.any(
            Dt
        )  # make sure at least one pair is available in every layer to be sampled

        u_hat = self.model.cA @ self.theta[-1]
        x_plus_y = u_hat + u_hat[:, None]
        # x_i = np.argmax(u_hat * Dt)
        # ii = np.arange(0, K, dtype=np.int32)[Dt]
        # y_i = self.rng.choice(ii)
        # return x_i, y_i

        # mask = Dt.reshape(-1, 1) * Dt.reshape(1, -1)
        Lvar = self.enorm
        cb = Lvar.min(axis=0)
        beta_tl = self.beta_tl_override()
        x_i, y_i = np.unravel_index(
            np.argmax((x_plus_y + beta_tl * cb), axis=None), (K, K)
        )
        # ii = np.arange(0, K, dtype=np.int32)[Dt]
        # y_i = self.rng.choice(ii)
        # print(x_i, y_i)
        return x_i, y_i


class MaxPairUCB2(MaxPairUCB):
    def beta_scale(self):
        return 0.5
