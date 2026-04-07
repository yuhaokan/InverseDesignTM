from envs import BilliardTwoEnv, BilliardThreeEnv
from load_pos import load_best_pos
import numpy as np
import meep as mp

# best_pos_BilliardTwo_Env12_Rank1_PPO_2
# best_pos_BilliardTwo_Env20_Rank1Trace0_PPO
# best_pos_BilliardTwo_Env12_DegenerateEigVal_PPO_8
# best_pos_BilliardTwo_Env12_DegenerateSingularVal_PPO
# best_pos_BilliardTwo_Env12_RM_Rank1_PPO
pos, error = load_best_pos(best_pos_file_name = 'best_pos_BilliardTwo_Env12_DegenerateEigVal_PPO_8.npy')

print(error)

env = BilliardTwoEnv()

############   Validate Rank-1
# tm = env._calculate_subSM(pos, matrix_type="TM", visualize=False)

# from envs import TargetType
# print(env._calculate_reward(tm, target_type=TargetType.DEGENERATE_EIG_VAL))

# normalized_tm = env._calculate_normalized_subSM(pos, matrix_type="TM", visualize=False)
# print(normalized_tm.T)
# _, s, _ = np.linalg.svd(tm)

# s = np.linalg.svd(normalized_tm, full_matrices=False, compute_uv=False)


# print(np.linalg.norm(normalized_tm, 'fro'))
# print(s)

# print(f'schmidt_number: {env.schmidt_number(normalized_tm)}')
# print(np.angle(tm[:,1]/tm[:,0]), np.abs(tm[:,1]/tm[:,0]))


# print(env._measure_incoming_amplitudes())

# env.plot_speckle_patterns_steady_state(pos, field_component=mp.Ez, input_port_index=0)

# env.plot_speckle_patterns_steady_state(pos, field_component=mp.Ez, input_port_index=1)

# env.plot_lowest_transmission_eigenchannel_steady_state(pos, field_component=mp.Ez, matrix_type="TM")

# env.plot_transmission_eigenvalue_spectrum(scatter_pos=pos, freq_range=(0.48, 0.52), freq_points=101, save_path='1')

# env.plot_schmidt_number_map(
#     scatter_pos=pos,
#     scatterer_index=5, 
#     direction='x',
#     position_range=(-0.1, 0.1),  # Small shifts around current position
#     position_steps=11,
#     freq_range=(0.49, 0.51),      # Narrow frequency range around design frequency
#     freq_steps=11,
#     save_path='1'
# )


# env.plot_phase_map(
#     scatter_pos=pos,
#     freq_range=(0.4995, 0.5005),  
#     freq_points=5,           
#     loss_range=(-1e-5, 1e-5), 
#     loss_points=5,            
#     # save_path='5'
# )

# env.plot_phase_map(
#     scatter_pos=pos,
#     freq_range=(0.496, 0.504),  
#     freq_points=5,           
#     loss_range=(-0.00001, 0.00005), 
#     loss_points=5,            
#     save_path=None
# )


#############   Validate degenerate eigenvalues

# normalized_tm = env._calculate_normalized_subSM(pos, matrix_type="TM", visualize=False)
# eigenvalues, elgenvectors = np.linalg.eig(normalized_tm)
# print(normalized_tm)
# print(eigenvalues)

# print(env.get_eigenvectors_degenerate_case(scatter_pos = pos))


# P = env.get_P(scatter_pos = pos)
# print(env.get_Jordan_near_EP(P, scatter_pos = pos))

# env.calculate_and_save_TM_sweep(
#     scatter_pos=pos,
#     scatter_idx=4,
#     position_range=(-0.0022, -0.00051),
#     position_points=51,
#     direction='x',
#     freq_range=(0.49972, 0.5001),           # (0.485, 0.515),
#     freq_points=1,
#     save_path='17'
# )

env.find_EP_sweep(
    scatter_pos=pos,
    scatter_idx=4,
    position_range=(-0.0028, -0.0021),
    position_points=11,
    direction='x',
    freq_range=(0.50004, 0.50008),           # (0.485, 0.515),
    freq_points=12,
    save_path='23'
)

# env.calculate_eigenvector_coalescence(
#     scatter_pos=pos,
#     freq_range=(0.485, 0.515),  
#     freq_points=51,           
#     loss_range=(-0.005, 0.01), 
#     loss_points=51,            
#     save_path=None
# )

# env.calculate_eigenvector_coalescence_position_sweep(
#     scatter_pos=pos,
#     scatter_idx=4,           # First scatterer
#     position_range=(-0.3, 0.3),  # Position perturbation range
#     position_points=51,  # Higher resolution in position space
#     direction='x',           # Perturb in x-direction
#     freq_range=(0.485, 0.515), # Frequency range to analyze
#     freq_points=51,          # Higher resolution for better detection of EPDs
#     save_path='0'
# )
