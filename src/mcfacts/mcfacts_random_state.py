
import numpy as np

default_seed = 1
rng = np.random.default_rng(default_seed)

def reset_random(seed):
	rng = np.random.default_rng(seed)

