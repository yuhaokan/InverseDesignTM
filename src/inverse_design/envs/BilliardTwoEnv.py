import numpy as np
import meep as mp
# from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
# from stable_baselines3.common.vec_env import DummyVecEnv

try:
    # Try relative i,port first (when used as part of the package)
    from .base_env import BilliardBaseEnv
except ImportError:
    # Fall back to direct import (when run as a script)
    from base_env import BilliardBaseEnv

# Suppress logging
mp.verbosity(0)

# register(
#     id='BilliardTwoEnv',                         
#     entry_point='BilliardTwoEnv:BilliardTwoEnv',        # module_name:class_name
# )


class BilliardTwoEnv(BilliardBaseEnv):
    def __init__(self):
        super().__init__()

        # Define ports
        self.source_ports = [
            {"name": "left_top", "position": mp.Vector3(-self.sx/2-self.source_billiard_distance, self.waveguide_offset), "direction": mp.X},
            {"name": "left_bottom", "position": mp.Vector3(-self.sx/2-self.source_billiard_distance, -self.waveguide_offset), "direction": mp.X}
        ]

        self.output_ports = [
            {"name": "right_top", "position": mp.Vector3(self.sx/2+self.source_billiard_distance, self.waveguide_offset), "direction": mp.X},
            {"name": "right_bottom", "position": mp.Vector3(self.sx/2+self.source_billiard_distance, -self.waveguide_offset), "direction": mp.X}
        ]

        self.reflection_ports = [
            {"name": "left_top", "position": mp.Vector3(-self.sx/2-self.source_billiard_distance/2, self.waveguide_offset), "direction": mp.X},
            {"name": "left_bottom", "position": mp.Vector3(-self.sx/2-self.source_billiard_distance/2, -self.waveguide_offset), "direction": mp.X}
        ]

    def _create_base_geometry(self):
        # Create the base geometry (billiard and waveguides)
        geometry = []
        
        # Left wall
        geometry.append(mp.Block(
            material=mp.perfect_electric_conductor,
            center=mp.Vector3(-self.sx/2-self.metal_thickness/2, 0),
            size=mp.Vector3(self.metal_thickness, self.sy+2*self.metal_thickness)
        ))

        # Right wall
        geometry.append(mp.Block(
            material=mp.perfect_electric_conductor,
            center=mp.Vector3(self.sx/2+self.metal_thickness/2, 0),
            size=mp.Vector3(self.metal_thickness, self.sy+2*self.metal_thickness)
        ))

        # Top wall
        geometry.append(mp.Block(
            material=mp.perfect_electric_conductor,
            center=mp.Vector3(0, self.sy/2+self.metal_thickness/2),
            size=mp.Vector3(self.sx+2*self.metal_thickness, self.metal_thickness)
        ))

        # Bottom wall
        geometry.append(mp.Block(
            material=mp.perfect_electric_conductor,
            center=mp.Vector3(0, -self.sy/2-self.metal_thickness/2),
            size=mp.Vector3(self.sx+2*self.metal_thickness, self.metal_thickness)
        ))
        
        # Create waveguides
        self._create_metal_waveguide(geometry, -self.sx/2-self.waveguide_length/2, self.waveguide_offset, 
                                    self.waveguide_length, self.waveguide_width)
        self._create_metal_waveguide(geometry, -self.sx/2-self.waveguide_length/2, -self.waveguide_offset, 
                                    self.waveguide_length, self.waveguide_width)
        self._create_metal_waveguide(geometry, self.sx/2+self.waveguide_length/2, self.waveguide_offset, 
                                    self.waveguide_length, self.waveguide_width)
        self._create_metal_waveguide(geometry, self.sx/2+self.waveguide_length/2, -self.waveguide_offset, 
                                    self.waveguide_length, self.waveguide_width)
        
        # Create openings
        # Left top opening
        geometry.append(mp.Block(
            material=mp.Medium(epsilon=self.epsilon_bg),
            center=mp.Vector3(-self.sx/2-self.metal_thickness/2, self.waveguide_offset),
            size=mp.Vector3(self.metal_thickness, self.waveguide_width)
        ))

        # Left bottom opening
        geometry.append(mp.Block(
            material=mp.Medium(epsilon=self.epsilon_bg),
            center=mp.Vector3(-self.sx/2-self.metal_thickness/2, -self.waveguide_offset),
            size=mp.Vector3(self.metal_thickness, self.waveguide_width)
        ))

        # Right top opening
        geometry.append(mp.Block(
            material=mp.Medium(epsilon=self.epsilon_bg),
            center=mp.Vector3(self.sx/2+self.metal_thickness/2, self.waveguide_offset),
            size=mp.Vector3(self.metal_thickness, self.waveguide_width)
        ))

        # Right bottom opening
        geometry.append(mp.Block(
            material=mp.Medium(epsilon=self.epsilon_bg),
            center=mp.Vector3(self.sx/2+self.metal_thickness/2, -self.waveguide_offset),
            size=mp.Vector3(self.metal_thickness, self.waveguide_width)
        ))

        # Fill main cavity with air
        geometry.append(mp.Block(
            material=mp.Medium(epsilon=self.epsilon_bg),
            center=mp.Vector3(0, 0),
            size=mp.Vector3(self.sx, self.sy)
        ))
        
        return geometry
       
    def _calculate_reward(self, tm) -> tuple[np.float32, np.float32]:
        # Target relationship: tm[0] * 1.73 = tm[1], expect a rank-1 TM
        ratio = np.sqrt(2)
        # error = np.sum((np.abs(tm[0] / tm[1]) * ratio - 1)**2) 

        # const power splitter
        # error = np.abs(tm[0][0] * ratio - tm[0][1]) + np.abs(tm[1][0] * ratio - tm[1][1])

        # rank-1 & trace-0
        # error = np.abs(tm[0][0] * tm[1][1] - tm[0][1] * tm[1][0]) + np.abs(tm[0][0] + tm[1][1])

        # error = np.mean(np.abs(tm[0] * ratio - tm[1]))
        error = np.abs(tm[0][0] * tm[1][1] - tm[0][1] * tm[1][0])

        # targetTM = np.array([[-2.28661274+0.54642883j, -7.33391126-0.31989986j], [4.91357518-2.36528964j,  3.44673878+3.01154595j]])
        # error = np.sum(np.abs(tm - targetTM))
        
        # Reward is negative of error (higher reward for lower error)
        reward = -error
        
        return reward, error
        

if __name__ == "__main__":
    env = BilliardTwoEnv()
    # # env = gym.make('MyEnv-v0')
    env.reset(seed=55)
    # print("check env begin")
    # check_env(env)
    # print("check env end")

    # normalized_rm = env._calculate_normalized_subSM(env.scatter_pos, matrix_type="RM", visualize=False)
    # normalized_tm = env._calculate_normalized_subSM(env.scatter_pos, matrix_type="TM", visualize=False)
    # print(normalized_rm @ np.conj(normalized_rm).T)
    # print(normalized_tm @ np.conj(normalized_tm).T)
    # print(normalized_rm @ np.conj(normalized_rm).T + normalized_tm @ np.conj(normalized_tm).T)

    tm_sample = env._calculate_subSM(env.scatter_pos, matrix_type="RM", visualize=False)
    # print(tm_sample)
    # print(env._calculate_reward(tm_sample))

    # env.render()

    # print(env.unwrapped.get_state())

    # env2 = DummyVecEnv([lambda: BilliardTwoEnv()])
    # print(env2.get_attr('n_scatterers'))
    
    # # Test episode loop
    # episodes = 2
    # for episode in range(episodes):
    #     obs, _ = env.reset(seed=episode)
    #     done = False
    #     truncated = False
    #     total_reward = 0
    #
    #     print(f"\nEpisode {episode + 1}")
    #     while not (done or truncated):
    #         action = env.action_space.sample()  # Replace with your action selection
    #         obs, reward, done, truncated, info = env.step(action)
    #         total_reward += reward
    #
    #         if episode == 0:  # Render only first episode
    #             env.render()
    #
    #     print(f"Episode {episode + 1} finished with reward: {total_reward}")
