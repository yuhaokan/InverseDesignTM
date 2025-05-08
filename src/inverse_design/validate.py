from envs import BilliardTwoEnv
from load_pos import load_best_pos
import numpy as np
import meep as mp

pos, error = load_best_pos(best_pos_file_name = 'best_pos_BilliardTwo_Env12_Rank1_PPO.npy')

print(error)

env = BilliardTwoEnv()

# tm = env._calculate_subSM(pos, matrix_type="TM", visualize=False)

# print(env._calculate_reward(tm))

# print(tm)

# print(tm[:,1]/tm[:,0])

# print(np.angle(tm[:,1]/tm[:,0]), np.abs(tm[:,1]/tm[:,0]))


# env.plot_lowest_transmission_eigenchannel(pos, field_component=mp.Ez)

env.plot_speckle_patterns(pos, field_component=mp.Ez)
