import os, sys
import time
from model import LinearLogitModel
from vdb import LCDB
from ucb import MaxInP, MaxFirstUCBNext, CoLSTIM, MaxFirstRndNext, MaxPairUCB
from suplin import AdaCDB, VACDB, StaD, StaDSim
from det import MaxScalarFishPair, ETC, MaxInpFish


class RND(MaxInP):
    def next_action(self):
        x_i = self.model.rng.integers(0, K)
        y_i = self.model.rng.integers(0, K)
        return x_i, y_i


if __name__ == "__main__":
    array_id = getattr(os.environ, "SLURM_ARRAY_TASK_ID", None)
    if len(sys.argv) == 2:
        seed = int(sys.argv[1])
    elif array_id:
        seed = int(array_id)
    else:
        seed = int(time.time() * 1000000)

    T = 4000
    d = 5
    K = 2**5
    alg_classes = [
        # RND,
        # MaxFirstMaxDet,
        # MaxDetGreedy,
        # StaticMaxDet,
        MaxInP,
        MaxFirstRndNext,
        MaxFirstUCBNext,
        MaxPairUCB,
        CoLSTIM,
    ]
    # why #L does not make difference?
    # the same pair contribute to the loss multiple times
    # large scale should be O(d)
    # scale best arm only
    # global estimator in combination w/ local estimator
    # colstim noise on/off compare with maxfirstucbnext
    # std'd supcolstim
    # cumulative sum of variance of algo
    todo_list = list(zip(alg_classes, [None] * len(alg_classes)))
    for l in range(5, 6):
        # todo_list.append((LCDB, l))
        # todo_list.append((AdaCDB, l))
        # todo_list.append((VACDB, l))
        # todo_list.append((StaD, l))
        pass
    for scale in [0.1, 0.5, 1, 2, 4][2:3]:
        for alg_cls, l in todo_list:
            model = LinearLogitModel(T, K, d, seed, scale=scale)
            if l:
                algo = alg_cls(T, model, L=l)
                suffix = f" L={l}"
            else:
                algo = alg_cls(T, model, L=1)
                suffix = ""
            suffix += f" scale={scale}"
            print(f"Starting with {alg_cls.__name__} {l if l else 0}..")
            algo.run()
            algo.summarize(suffix)
            print(f"Finished with {alg_cls.__name__}.")
