from abc import ABC, abstractmethod
import numpy as np

from model import Model


class DB(ABC):
    def __init__(self, T: int, model: Model, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.model = model
        self.T = T

        self.error = []

    def run(self):
        """
        main loop
        """
        if self.L:
            self.error = [[] for _ in range(self.L + 1)]

        for t in range(1, 1 + self.T):
            # print(f"{t}", end="\r")
            self.t = t
            # model can also change
            act = self.next_action()
            # print(act)
            r = self.model.action(t, act)
            self.estimate(r, act)
            self.record_error()

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

    def record_error(self):
        theta = self.theta
        if self.L:
            for l in range(self.L + 1):
                self.error[l].append(np.linalg.norm(self.model.theta_star - theta[l]))
        else:
            self.error.append(np.linalg.norm(self.model.theta_star - theta))
