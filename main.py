import sys
from model import LinearLogitModel
from vdb import VDBGLM
from ucb import MaxInp, MaxFirstUCBNext, MaxFirstRndNext, MaxPairUCB
from suplin import AdaDBGLM, SAVE


class RND(VDBGLM):
    def next_action(self):
        x_i = self.rng.integers(0, K)
        y_i = self.rng.integers(0, K)
        return x_i, y_i


if __name__ == "__main__":
    if len(sys.argv) == 2:
        seed = int(sys.argv[1])
    else:
        import time

        seed = time.time() * 1000000
        seed = int(seed)
        # seed = 12345
    print("seed", seed)

    # for eta_scale in np.arange(0.004, 0.011, 0.001):
    T = 4000
    d = 5
    K = 2**5
    alg_classes = [
        # RND,
        # MaxFirstMaxDet,
        # MaxDetGreedy,
        # StaticMaxDet,
        MaxInp,
        MaxFirstRndNext,
        MaxFirstUCBNext,
        # MaxFirstRowMaxNext,
        MaxPairUCB,
        # VDBGLM,
        # AdaDBGLM,
        # SAVE,
    ]
    todo_list = list(zip(alg_classes, [None] * len(alg_classes)))
    for l in range(3, 9):
        todo_list.append((SAVE, l))
    for scale in [0.1, 0.5, 1, 2, 4]:
        for alg_cls, l in todo_list:
            model = LinearLogitModel(T, K, d, seed, scale=scale)
            if l:
                algo = alg_cls(T, model, seed, L=l)
                suffix = f" L={l}"
            else:
                algo = alg_cls(T, model, seed)
                suffix = ""
            suffix += f" scale={scale}"
            print(f"Starting with {alg_cls.__name__}..")
            algo.run()
            algo.summarize(suffix)
            print(f"Finished with {alg_cls.__name__}.")
