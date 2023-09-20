from abc import ABC, abstractmethod
import numpy as np

from model import Model, DTYPE


class DB(ABC):
    """
    abstract class for Dueling Bandit
    """

    def __init__(self, T: int, model: Model, L: int) -> None:
        self.T = T
        self.delta = 1.0 / self.T
        self.model = model
        self.error = None

        self.d = model.d
        self.K = model.K
        d = self.d
        K = self.K

        self.g_z = self.model.g_z
        self.g_z_outer = self.model.g_z_outer

        # L = int(np.ceil(np.log2(1.0 / 2 / self.alpha)))
        # quick hack for non layered methods
        if self.single_layer():
            L = 1
        self.L = L
        self.alpha = 1.0 / (2**L)

        # index 0 is global estimator
        # index [1, L] inclusive is layerd estimators
        # if L == 1 then local and global estimator should be the same
        self.enorm = np.ones((L + 1, K, K), dtype=DTYPE)
        self.z = [np.zeros((0, d), dtype=DTYPE) for _ in range(L + 1)]
        self.r = [np.array([], dtype=DTYPE) for _ in range(L + 1)]
        self.w = [np.array([], dtype=DTYPE) for _ in range(L + 1)]
        self.D = np.ones((L + 1, K), dtype=DTYPE)  # feasible set
        self.Psi = np.zeros((L + 1))  # sample count in each layer
        self.count_xy = np.zeros((K, K))
        self.count_xyL = np.zeros((L + 1, K, K))

        self.init_sigma()
        # init enorm
        for l in range(L + 1):
            self.enorm[l] = np.sqrt(
                (self.g_z.reshape(K, K, 1, d) @ self.SigmaInv[l])
                @ self.g_z.reshape(K, K, d, 1)
            ).reshape(K, K)

        # estimated theta
        self.theta = np.zeros((L + 1, d)) / d

    @abstractmethod
    def init_sigma(self):
        pass

    def run(self):
        """main loop for T rounds"""
        # for single layer, L = 1 + 1, for L layers self.L = L + 1
        # L = 0 denotes the global estimator
        self.error = [[] for _ in range(self.L + 1)]

        for t in range(1, 1 + self.T):
            # print(f"{t}", end="\r")
            self.t = t
            # model can also change
            act = self.next_action()
            # print(act)
            r = self.model.action(t, act)
            self.estimate(r, act)
            self.record_error(act)

    @abstractmethod
    def next_action(self):
        """
        according to the current estimate, using predefined strategy generate a new query/action to perform
        """
        pass

    @abstractmethod
    def estimate(self, r):
        """
        updates model estimation based on new observation and the past
        """
        pass

    def record_error(self, act):
        theta = self.theta
        x_i, y_i = act
        z = self.model.cA[x_i] - self.model.cA[y_i]
        self.error[0].append(self.model.mu(self.model.theta_star @ z))

        for l in range(1, self.L + 1):
            self.error[l].append(
                np.linalg.norm(self.model.theta_star - theta[l]) / self.model.scale
            )
