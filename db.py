from abc import ABC, abstractmethod
import numpy as np

from model import Model


class DB(ABC):
    def __init__(self, T: int, model: Model, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.model = model
        self.T = T

    def run(self):
        """
        main loop
        """
        for t in range(1, 1 + self.T):
            # print(f"{t}", end="\r")
            self.t = t
            # model can also change
            act = self.next_action()
            # print(act)
            r = self.model.action(t, act)
            self.estimate(r, act)
            self.model.record_error(self.theta)
            # self.model.record_error(self.theta[-1])

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
