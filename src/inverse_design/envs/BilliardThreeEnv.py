import numpy as np
import meep as mp
from gymnasium.utils.env_checker import check_env

try:
    # Try relative i,port first (when used as part of the package)
    from .base_env import BilliardBaseEnv
except ImportError:
    # Fall back to direct import (when run as a script)
    from base_env import BilliardBaseEnv

# Suppress logging
mp.verbosity(0)

class BilliardThreeEnv(BilliardBaseEnv):
    def __init__(self):
        super().__init__()

        # Define ports - now with 3 input and 3 output ports (added center ports)
        self.source_ports = [
            {"name": "left_top", "position": mp.Vector3(-self.sx/2-self.source_billiard_distance, self.waveguide_offset), "direction": mp.X},
            {"name": "left_center", "position": mp.Vector3(-self.sx/2-self.source_billiard_distance, 0), "direction": mp.X},
            {"name": "left_bottom", "position": mp.Vector3(-self.sx/2-self.source_billiard_distance, -self.waveguide_offset), "direction": mp.X}
        ]

        self.output_ports = [
            {"name": "right_top", "position": mp.Vector3(self.sx/2+self.source_billiard_distance, self.waveguide_offset), "direction": mp.X},
            {"name": "right_center", "position": mp.Vector3(self.sx/2+self.source_billiard_distance, 0), "direction": mp.X},
            {"name": "right_bottom", "position": mp.Vector3(self.sx/2+self.source_billiard_distance, -self.waveguide_offset), "direction": mp.X}
        ]

        self.reflection_ports = [
            {"name": "left_top", "position": mp.Vector3(-self.sx/2-self.source_billiard_distance/2, self.waveguide_offset), "direction": mp.X},
            {"name": "left_center", "position": mp.Vector3(-self.sx/2-self.source_billiard_distance/2, 0), "direction": mp.X},
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
        # Left waveguides
        self._create_metal_waveguide(geometry, -self.sx/2-self.waveguide_length/2, self.waveguide_offset, 
                                    self.waveguide_length, self.waveguide_width)
        self._create_metal_waveguide(geometry, -self.sx/2-self.waveguide_length/2, 0, 
                                    self.waveguide_length, self.waveguide_width)
        self._create_metal_waveguide(geometry, -self.sx/2-self.waveguide_length/2, -self.waveguide_offset, 
                                    self.waveguide_length, self.waveguide_width)
        
        # Right waveguides
        self._create_metal_waveguide(geometry, self.sx/2+self.waveguide_length/2, self.waveguide_offset, 
                                    self.waveguide_length, self.waveguide_width)
        self._create_metal_waveguide(geometry, self.sx/2+self.waveguide_length/2, 0, 
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

        # Left center opening
        geometry.append(mp.Block(
            material=mp.Medium(epsilon=self.epsilon_bg),
            center=mp.Vector3(-self.sx/2-self.metal_thickness/2, 0),
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

        # Right center opening
        geometry.append(mp.Block(
            material=mp.Medium(epsilon=self.epsilon_bg),
            center=mp.Vector3(self.sx/2+self.metal_thickness/2, 0),
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
        """
        Calculate reward based on how close the transmission matrix is to being rank-1.
        For a rank-1 matrix, the singular values beyond the first one should be zero.
        """
        # Convert the transmission matrix to a numpy array
        tm_array = np.array(tm)
        
        # Calculate the singular values of the matrix
        singular_values = np.linalg.svd(tm_array, compute_uv=False)
        
        # For a perfect rank-1 matrix, all singular values except the first should be zero
        # So we sum the squares of all singular values after the first one
        # error = np.sum(singular_values[1:]**2)
        
        # Alternatively, we can use the ratio of the first singular value to the sum
        # This measures how much of the matrix's "energy" is in the first singular value
        ratio = singular_values[0] / np.sum(singular_values)
        error = 1 - ratio  # Error is small when ratio is close to 1
        
        # Reward is negative of error (higher reward for lower error)
        reward = -error
        
        return reward, error

if __name__ == "__main__":
    env = BilliardThreeEnv()

    env.reset(seed=55)
    # print("check env begin")
    # check_env(env)
    # print("check env end")

    rm = env.calculate_normalized_subSM(env.scatter_pos, matrix_type="RM", visualize=False)
    tm = env.calculate_normalized_subSM(env.scatter_pos, matrix_type="TM", visualize=False)
    # print(tm)
    # print(env._calculate_reward(tm))

    print(rm @ np.conj(rm).T + tm @ np.conj(tm).T)
