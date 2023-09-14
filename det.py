import numpy as np
from db import DB
from model import Model
from ucb import MaxInp

dtype = np.float64
import itertools


def generate_combinations(vector, r):
    combinations = list(itertools.combinations(vector, r))
    return combinations


class MaxFirstMaxDet(MaxInp):
    def next_action(self) -> None:
        K = self.K
        Dt = self.D.sum(axis=0) == self.L
        assert np.any(
            Dt
        )  # make sure at least one pair is available in every layer to be sampled
        x_i, y_i = super().next_action()
        u_hat = self.model.cA @ self.theta[-1]
        x_i = np.argmax(u_hat * Dt)

        sigma_g = self.model.mu(self.g_z @ self.theta[0])  # K x K
        scale_g = sigma_g * (1 - sigma_g)
        fim = np.sum(
            (scale_g * self.count).reshape(K, K, 1, 1) * self.g_z_outer, axis=(0, 1)
        )
        det = np.linalg.det(fim)

        if det > 0:
            max_det = det
            # x_i, y_i = (0, 0)
            for i in [x_i]:
                for j in range(i, K):
                    if not (Dt[i] and Dt[j]):
                        continue
                    det_ij = np.linalg.det(fim + scale_g[i, j] * self.g_z_outer[i, j])
                    if det_ij > max_det:
                        x_i, y_i = (i, j)
                        max_det = det_ij

        self.count[x_i, y_i] += 1
        self.count[y_i, x_i] += 1
        return x_i, y_i


class MaxDetGreedy(MaxInp):
    def next_action(self) -> None:
        K = self.K
        Dt = self.D.sum(axis=0) == self.L
        assert np.any(
            Dt
        )  # make sure at least one pair is available in every layer to be sampled
        x_i, y_i = super().next_action()

        sigma_g = self.model.mu(self.g_z @ self.theta[0])  # K x K
        scale_g = sigma_g * (1 - sigma_g)
        fim = np.sum(
            (scale_g * self.count).reshape(K, K, 1, 1) * self.g_z_outer, axis=(0, 1)
        )
        det = np.linalg.det(fim)

        if det > 0:
            max_det = det
            for i in range(K):
                for j in range(i, K):
                    if not (Dt[i] and Dt[j]):
                        continue
                    det_ij = np.linalg.det(fim + scale_g[i, j] * self.g_z_outer[i, j])
                    if det_ij > max_det:
                        x_i, y_i = (i, j)
                        max_det = det_ij

        self.count[x_i, y_i] += 1
        self.count[y_i, x_i] += 1
        return x_i, y_i


class StaticMaxDet(MaxInp):
    def __init__(self, T: int, model: Model, seed: int) -> None:
        super().__init__(T, model, seed)
        self.find_max_comb()
        self.prev_Dt = np.ones(self.model.K, dtype=np.bool_)

    def find_max_comb(self):
        n_sel_pairs = d
        pairs_idx = generate_combinations(np.arange(0, K, 1, dtype=np.int32), 2)
        n_pairs = len(pairs_idx)
        combs = generate_combinations(
            np.arange(0, n_pairs, 1, dtype=np.int32), n_sel_pairs
        )
        max_det = 0
        max_comb = np.zeros(n_pairs)
        # all_dets = []
        for ci in range(len(combs)):
            I = np.zeros(d)
            comb = combs[ci]
            for pi in range(n_sel_pairs):
                i, j = pairs_idx[comb[pi]]
                xt = self.model.cA[i, :]
                yt = self.model.cA[j, :]
                z = xt - yt
                I_this_pair = (
                    self.model.mu(z @ self.model.theta_star)
                    * self.model.mu(-z @ self.model.theta_star)
                    * np.outer(z, z)
                )
                I = I + I_this_pair
            this_det = np.linalg.det(I)
            # % all_dets = [all_dets this_det]
            if this_det >= max_det:
                max_det = this_det
                max_comb = comb
            print(int(ci / len(combs) * 100), end="\r")
        # save info for sampling usage
        self.pairs_idx = pairs_idx
        self.max_comb = max_comb

    def next_action(self):
        Dt = self.D.sum(axis=0) == self.L
        assert np.any(
            Dt
        )  # make sure at least one pair is available in every layer to be sampled
        if np.sum(Dt) > d:
            if not np.all(self.prev_Dt == Dt):
                self.find_max_comb()
        else:
            return MaxFirstRndNext.next_action(self)

        pair_i = self.rng.choice(self.max_comb)
        return self.pairs_idx[pair_i]
