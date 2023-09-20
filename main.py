import os, sys
import time
from model import LinearLogitModel
from vdb import LCDB
from ucb import MaxInp, MaxFirstUCBNext, MaxFirstRndNext, MaxPairUCB
from suplin import AdaCDB, StaAdaCDB


class RND(LCDB):
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

    T = 2000
    d = 5
    K = 2**5
    alg_classes = [
        # RND,
        # MaxFirstMaxDet,
        # MaxDetGreedy,
        # StaticMaxDet,
        MaxInp,
        # MaxFirstRndNext,
        # MaxFirstUCBNext,
        MaxPairUCB,
    ]
    todo_list = list(zip(alg_classes, [None] * len(alg_classes)))
    for l in range(1, 4):
        # todo_list.append((LCDB, l))
        # todo_list.append((AdaCDB, l))
        todo_list.append((StaAdaCDB, l))
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
