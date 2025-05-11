from envs import BilliardTwoEnv, BilliardThreeEnv
from load_pos import load_best_pos
import numpy as np
import meep as mp

# best_pos_BilliardTwo_Env12_Rank1_PPO
pos, error = load_best_pos(best_pos_file_name = 'best_pos_BilliardTwo_Env12_DegenerateEigVal_PPO_1.npy')

print(error)

env = BilliardTwoEnv()

############   Validate Rank-1
# tm = env._calculate_subSM(pos, matrix_type="TM", visualize=False)

# print(env._calculate_reward(tm, target_type = "DegenerateEigVal"))

# print(tm)

# print(tm[:,1]/tm[:,0])

# print(np.angle(tm[:,1]/tm[:,0]), np.abs(tm[:,1]/tm[:,0]))


# env.plot_lowest_transmission_eigenchannel(pos, field_component=mp.Ez)


# env.plot_speckle_patterns(pos, field_component=mp.Ez)

# env.plot_phase_map(
#     scatter_pos=pos,
#     freq_range=(0.4998, 0.5002),  
#     freq_points=5,           
#     loss_range=(-0.000005, 0.000005), 
#     loss_points=5,            
#     save_path=None
# )


#############   Validate degenerate eigenvalues

# normalized_tm = env._calculate_normalized_subSM(pos, matrix_type="TM", visualize=False)
# eigenvalues, elgenvectors = np.linalg.eig(normalized_tm)
# print(eigenvalues)

# env.calculate_eigenvector_coalescence(
#     scatter_pos=pos,
#     freq_range=(0.485, 0.515),  
#     freq_points=51,           
#     loss_range=(-0.005, 0.01), 
#     loss_points=51,            
#     save_path=None
# )