from envs import BilliardTwoEnv
from load_pos import load_best_pos


pos, error = load_best_pos()

env = BilliardTwoEnv()
env.reset(seed=55)
tm = env._calculate_tm(pos)

print(tm)

print(tm[0] * 1.73)

print(tm[0] * 1.73 - tm[1])

