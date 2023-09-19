from abc import ABC, abstractmethod
import numpy as np
import scipy.special
import scipy.spatial
import scipy.optimize

dtype = np.float64


class Model(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def action(self, act):
        raise NotImplementedError

    # @abstractmethod
    # def step(self):
    #     """
    #     sometimes the environment changes adversarily
    #     """
    #     raise NotImplementedError


class LinearLogitModel(Model):
    def __init__(self, T, K, d, seed, scale=1) -> None:
        super().__init__()

        self.T = T
        self.K = K
        self.d = d
        self.seed = seed
        self.data_rng = np.random.default_rng(23333)
        self.rng = np.random.default_rng(seed)
        self.mu = scipy.special.expit
        # self.R = np.zeros((T + 1), dtype=dtype)
        self.R = []
        self.scale = scale

        # generate a random ground truth parameter and normalize it
        theta_star = self.data_rng.random((d,), dtype=dtype)
        # theta_star = (theta_star > 0.5).astype(np.int64) * 2.0 - 1

        self.theta_star = theta_star / np.sqrt(theta_star @ theta_star)
        self.theta_star *= self.scale
        # print(self.theta_star)

        # generate feature vectors for arms
        # it is random binary vectors with disturbance
        x_orig = self.data_rng.integers(0, 2**d, size=(K,), dtype=np.int32)  # [0, 2^d)
        x_orig = np.arange(K)
        # cA[0:K//4] += rng.random((K//4, d)) * 2

        # norm_samples = self.rng.normal(size=(1, d))
        # norm_samples /= np.sqrt(np.sum(norm_samples ** 2, axis=1))
        # rad_1 = 0
        # rad_2 = 1. / np.sqrt(d)
        # scale = (self.rng.uniform(rad_1, rad_2, size=(1, )) ** (1./d))
        # self.theta_star = (norm_samples * scale).reshape(d)

        cA = (
            (x_orig.reshape(-1, 1) & (2 ** np.arange(d))) != 0
        ) * 2.0 - 1.0  # convert to binary vectors
        # print(cA)
        # cA += (self.rng.random(cA.shape) - 1) / 10  # add some disturb with mean 0
        # cA = self.rng.random((K, d)) - 0.5 # completely random
        self.cA = cA
        # self.cA = self.rng.uniform(0, 1, size=(K, d))

        g_z = cA.reshape(K, 1, d) - cA.reshape(1, K, d)
        self.g_z_outer = g_z.reshape(K, K, d, 1) @ g_z.reshape(K, K, 1, d)
        self.p = self.mu(g_z @ self.theta_star)
        self.g_z = g_z
        # sanity check
        # for i in range(K):
        #     for j in range(K):
        #         assert np.all(g_z[i, j] == cA[i] - cA[j])
        #         assert np.all(
        #             g_z_outer[i, j] == np.outer(cA[i] - cA[j], cA[i] - cA[j])
        #         )
        #         assert (p[i, j] - mu(cA[i] @ theta_star - cA[j] @ theta_star)) < 1e-10
        # print(p)
        # print(cA)
        # print(theta_star)

        u = self.mu(self.cA @ self.theta_star)
        self.x_star_idx = np.argmax(u)

    def action(self, t, act):
        x_i, y_i = act
        r = self.rng.binomial(1, self.p[x_i, y_i])
        x = self.cA[x_i]
        y = self.cA[y_i]

        # if the regret depends on link function the strategy can be differ?
        # lipschtizness
        # self.R[t] = (2 * self.cA[self.x_star_idx] - x - y) @ self.theta_star + self.R[
        #     t - 1
        # ]
        R_t = (2 * self.cA[self.x_star_idx] - x - y) @ self.theta_star / self.scale
        if len(self.R):
            self.R.append(R_t + self.R[-1])
        else:
            self.R.append(R_t)
        return r
