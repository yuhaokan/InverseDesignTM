from envs import BilliardTwoEnv
from load_pos import load_best_pos
import numpy as np
import meep as mp

# best_pos_BilliardTwo_Env12_Rank1_PPO
pos, error = load_best_pos(best_pos_file_name = 'best_pos_BilliardTwo_Env12_Rank1_PPO.npy')

print(error)

env = BilliardTwoEnv()

# tm = env._calculate_subSM(pos, matrix_type="TM", visualize=False)

# print(env._calculate_reward(tm))

# print(tm)

# print(tm[:,1]/tm[:,0])

# print(np.angle(tm[:,1]/tm[:,0]), np.abs(tm[:,1]/tm[:,0]))


# print(env._calculate_normalized_subSM(pos, matrix_type="TM", visualize=False))


# env.plot_lowest_transmission_eigenchannel(pos, field_component=mp.Ez)


# env.plot_speckle_patterns(pos, field_component=mp.Ez)

# env.plot_phase_map(
#     scatter_pos=pos,
#     freq_range=(0.4998, 0.5002),      # 90% to 110% of base frequency
#     freq_points=21,             # 21 frequency points
#     loss_range=(-0.000005, 0.000005),   # Loss factors from 0.001 to 0.05
#     loss_points=21,             # 21 loss factor points
#     save_path=None   # Save the figure
# )