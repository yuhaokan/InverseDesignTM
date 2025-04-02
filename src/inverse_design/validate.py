from envs import BilliardTwoEnv
from load_pos import load_best_pos
import numpy as np

pos, error = load_best_pos()

print(error)

env = BilliardTwoEnv()
# env.reset(seed=55)
tm = env._calculate_tm(pos)

print(tm)

print(tm[:,1]/tm[:,0])

print(np.angle(tm[:,1]/tm[:,0]), np.abs(tm[:,1]/tm[:,0]))

