import sys
from model import LinearLogitModel
from vdb import VDBGLM
from ucb import MaxInp, MaxFirstUCBNext, MaxFirstRndNext
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
        seed = 12345
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
        #MaxInp,
        # MaxFirstRndNext,
        # MaxFirstUCBNext,
        # MaxFirstRowMaxNext,
        # MaxFirstRndNextNoMask,
        # MaxPairUCB2,
        # MaxPairUCB,
        #VDBGLM,
        #AdaDBGLM,
        SAVE,
    ]
    for alg_cls in alg_classes:
        model = LinearLogitModel(T, K, d, seed)
        algo = alg_cls(T, model, seed)
        algo.run()
        print(f"Starting with {alg_cls.__name__}...")
        algo.summarize()
        print(f"Finished with {alg_cls.__name__}.")
