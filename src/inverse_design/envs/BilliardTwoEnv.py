import numpy as np
import meep as mp
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import typing
from stable_baselines3.common.vec_env import DummyVecEnv

# Suppress logging
mp.verbosity(0)

# register(
#     id='BilliardTwoEnv',                         
#     entry_point='BilliardTwoEnv:BilliardTwoEnv',        # module_name:class_name
# )


class BilliardTwoEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.max_step = 1e5  ############################## for each episode, max steps we allowed
        
        self.n_scatterers = 20
        # Define action and observation spaces

        # both the action & obs space = n_scatterers * n_dim
        self.action_space = spaces.Box(low=-1, high=1, shape=(2 * self.n_scatterers,), dtype=np.float32) 

        # normalized coordinates, we should normalize the position by (length - 2*scatterer_radius)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2 * self.n_scatterers,), dtype=np.float32) 
        
        # MEEP simulation parameters
        self.resolution = 15  # pixels/cm

        '''
        a = 0.01  chosen characteristic length = 1cm
        c = 3e8   speed of light
        target_freq_GHz = 15.0
        fsrc = target_freq_GHz * 1e9 * a / c = target_freq_GHz / 30
        '''
        self.fsrc = 15.0 / 30

        self.sx = 15
        self.sy = 15
        self.scatterer_radius = 0.5

        self.sx_scatterer = self.sx - 2 * self.scatterer_radius # range of scatterer center 
        self.sy_scatterer = self.sy - 2 * self.scatterer_radius

        self.waveguide_width = 1.2
        self.waveguide_length = 8.0
        self.waveguide_offset = 3.0  # waveguide center distance to billiard horizontal midline

        self.pml_thickness = 3
        self.metal_thickness = 0.2

        self.epsilon_bg = 1.0
        self.epsilon_scatter = 3.9
        
        self.mode_num = 1
        
        # Define ports
        self.source_ports = [
            {"name": "left_top", "position": mp.Vector3(-self.sx/2-self.waveguide_length+self.pml_thickness+1, self.waveguide_offset), "direction": mp.X},
            {"name": "left_bottom", "position": mp.Vector3(-self.sx/2-self.waveguide_length+self.pml_thickness+1, -self.waveguide_offset), "direction": mp.X}
        ]

        self.output_ports = [
            {"name": "right_top", "position": mp.Vector3(self.sx/2+self.waveguide_length-self.pml_thickness-1, self.waveguide_offset), "direction": mp.X},
            {"name": "right_bottom", "position": mp.Vector3(self.sx/2+self.waveguide_length-self.pml_thickness-1, -self.waveguide_offset), "direction": mp.X}
        ]

        # Meep use the 'last object wins' principle, ie, if multiple objects overlap, later objects in the list take precedence. 
        # Allow overlapping simplifies implementation and help explore more diverse configurations.
        # Initial scatterer positions, this is normalized position !!!
        self.scatter_pos = self._generate_initial_positions()
        
        # For tracking progress
        self.best_error = float('inf')
        self.best_positions = None
        self.step_count = 0


    def _generate_initial_positions(self, seed=None):
        # Generate and normalize random positions
        # Use self.np_random instead of np.random to ensure proper seeding
        # self.np_random is provided by gym.Env and is properly seeded during reset()
        return self.np_random.uniform(low=-1, high=1, size=(2*self.n_scatterers,)).astype(np.float32)

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
    
    def _create_metal_waveguide(self, geometry, x_center, y_center, length, width):
        # Top wall of waveguide
        geometry.append(mp.Block(
            material=mp.perfect_electric_conductor,
            center=mp.Vector3(x_center, y_center + width/2 + self.metal_thickness/2),
            size=mp.Vector3(length, self.metal_thickness)
        ))
       
        # Bottom wall of waveguide
        geometry.append(mp.Block(
            material=mp.perfect_electric_conductor,
            center=mp.Vector3(x_center, y_center - width/2 - self.metal_thickness/2),
            size=mp.Vector3(length, self.metal_thickness)
        ))
       
        # Air inside the waveguide
        geometry.append(mp.Block(
            material=mp.Medium(epsilon=self.epsilon_bg),
            center=mp.Vector3(x_center, y_center),
            size=mp.Vector3(length, width)
        ))
    
    def _create_full_geometry(self, scatter_positions):
        # Convert normalized positions to actual coordinates
        actual_positions = []
        for i in range(0, 2 * self.n_scatterers, 2):
            x = scatter_positions[i] * (self.sx_scatterer/2)
            y = scatter_positions[i+1] * (self.sy_scatterer/2)
            actual_positions.append((x, y))

        # Create base geometry
        geometry = self._create_base_geometry()
        
        # Add scatterers
        for pos in actual_positions:
            geometry.append(mp.Cylinder(
                radius=self.scatterer_radius,
                height=0,
                center=mp.Vector3(pos[0], pos[1]),
                material=mp.Medium(epsilon=self.epsilon_scatter)
            ))
            
        return geometry
    
    def _run_simulation_for_port(self, input_port, geometry, run=True):
        # Create a new simulation for this port
        cell_size = mp.Vector3(self.sx + 2*self.waveguide_length, self.sy + 2*self.metal_thickness)
        pml_layers = [mp.PML(self.pml_thickness, direction=mp.X)]
        
        sources = [mp.EigenModeSource(
            mp.ContinuousSource(frequency=self.fsrc),
            center=input_port["position"],
            size=mp.Vector3(0, self.waveguide_width-0.1),
            eig_band=self.mode_num,
            eig_parity=mp.EVEN_Z + mp.ODD_Y  # TE mode
        )]
        
        sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=sources,
            resolution=self.resolution,
            dimensions=2
        )
        
        # Add monitors
        mode_monitor_right_top = sim.add_mode_monitor(
            self.fsrc, 0, 1,
            mp.ModeRegion(
                center=mp.Vector3(self.sx/2+self.waveguide_length-self.pml_thickness-1, self.waveguide_offset),
                size=mp.Vector3(0, self.waveguide_width-0.1)
            )
        )

        mode_monitor_right_bottom = sim.add_mode_monitor(
            self.fsrc, 0, 1,
            mp.ModeRegion(
                center=mp.Vector3(self.sx/2+self.waveguide_length-self.pml_thickness-1, -self.waveguide_offset),
                size=mp.Vector3(0, self.waveguide_width-0.1)
            )
        )

        # # monitor input
        # mode_monitor_left_top = sim.add_mode_monitor(
        #     self.fsrc, 0, 1,
        #     mp.ModeRegion(
        #         center=mp.Vector3(-self.sx/2-self.waveguide_length+self.pml_thickness+3, self.waveguide_offset),
        #         size=mp.Vector3(0, self.waveguide_width-0.1)
        #     )
        # )

        # mode_monitor_left_bottom = sim.add_mode_monitor(
        #     self.fsrc, 0, 1,
        #     mp.ModeRegion(
        #         center=mp.Vector3(-self.sx/2-self.waveguide_length+self.pml_thickness+3, -self.waveguide_offset),
        #         size=mp.Vector3(0, self.waveguide_width-0.1)
        #     )
        # )
        
        if not run:
            plt.figure()
            sim.plot2D(plot_eps_flag=True)
            return []

        else:    
            # Run simulation
            sim.run(until=100)  # Reduced simulation time for RL iterations
            
            # Calculate transmission coefficients
            mode_data_top = sim.get_eigenmode_coefficients(
                mode_monitor_right_top, [1],
                eig_parity=mp.EVEN_Z + mp.ODD_Y
            )
            
            mode_data_bottom = sim.get_eigenmode_coefficients(
                mode_monitor_right_bottom, [1],
                eig_parity=mp.EVEN_Z + mp.ODD_Y
            )

            # mode_data_input_top = sim.get_eigenmode_coefficients(
            #     mode_monitor_left_top, [1],
            #     eig_parity=mp.EVEN_Z + mp.ODD_Y
            # )
            
            # mode_data_input_bottom = sim.get_eigenmode_coefficients(
            #     mode_monitor_left_bottom, [1],
            #     eig_parity=mp.EVEN_Z + mp.ODD_Y
            # )
            
            # print('input strength')
            # print(mode_data_input_top.alpha[0,0,0])
            # print(mode_data_input_bottom.alpha[0,0,0])

            t_11 = mode_data_top.alpha[0,0,0]
            t_12 = mode_data_bottom.alpha[0,0,0]
            
            return [t_11, t_12]
    
    def _calculate_tm(self, normalized_scatterers_positions):
        
        # Create geometry with scatterers
        geometry = self._create_full_geometry(normalized_scatterers_positions)
        
        # Calculate transmission matrix
        t_matrix = []
        for input_port in self.source_ports:
            s_params = self._run_simulation_for_port(input_port, geometry)
            t_matrix.append(s_params)
        
        return np.array(t_matrix)
    
    def _calculate_reward(self, tm) -> tuple[np.float32, np.float32]:
        # Target relationship: tm[0] * 1.73 = tm[1], expect a rank-1 TM
        ratio = 1.41
        # error = np.sum((np.abs(tm[0] / tm[1]) * ratio - 1)**2) 

        # error = np.mean(np.abs(tm[0] * ratio - tm[1]))
        error = np.abs(tm[0][0] * tm[1][1] - tm[0][1] * tm[1][0])
        
        # Reward is negative of error (higher reward for lower error)
        reward = -error
        
        return reward, error

    def step(self, action) -> tuple[spaces.Box, np.float32, bool, bool, dict[str, typing.Any]]:
        self.step_count += 1

        # Apply action (small adjustments to positions)
        scaling_factor = 0.001  # Control adjustment size
        self.scatter_pos = np.clip(self.scatter_pos + action * scaling_factor, -1, 1)
        
        # Calculate transmission matrix with new positions
        tm = self._calculate_tm(self.scatter_pos)
        
        # Calculate reward and error
        reward, error = self._calculate_reward(tm)
        
        # Check if goal is achieved or max steps reached
        terminated = error < 0.1         #  5% deviation for 1.73*t11 vs t21, 5% deviation for 1.73*t12 vs t22, error_threshold = 2 * (5%)^2 = 0.005
        
        truncated = self.step_count >= self.max_step

        info = {
            "error": error,
            "scatter_pos": self.scatter_pos,
            "step": self.step_count
        }
        
        # Print progress every 10 steps
        if self.step_count % 10 == 0:
            print(f"Step {self.step_count}, Error: {error:.6f}, Reward: {reward:.6f}")

        return self.scatter_pos, reward, terminated, truncated, info

    # at the beginning of each episode, reset env
    def reset(self, seed=None, options=None) -> tuple[spaces.Box, dict[str, typing.Any]]:
        """
        Reset the environment.

        Args:
            seed (int, optional): Random seed for reproducibility
            options: None

        Returns:
            tuple: (observation, info_dict)
        """
        # Important: Call super().reset() first to properly seed the environment
        super().reset(seed=seed)

        # Optionally reset to best known positions with small perturbation
        if self.best_positions is not None and self.np_random.random() < 0.7:
            # 70% chance to use best positions with small Gaussian noise
            noise = self.np_random.normal(0, 0.05, size=(2*self.n_scatterers,)).astype(np.float32)
            self.scatter_pos = np.clip(self.best_positions + noise, -1, 1)
        else:
            # 30% chance to generate new random positions
            self.scatter_pos = self._generate_initial_positions(seed)

        self.scatter_pos = self._generate_initial_positions()

        self.step_count = 0

        # Create info dictionary
        info = {}

        # Return both observation and info dict
        return self.scatter_pos, info

    def get_state(self) -> dict[str, typing.Any]:
        """Get the current state of the environment."""
        return {
            'scatter_pos': self.scatter_pos.copy(),
            'step_count': self.step_count,
        }

    def set_state(self, state: dict[str, typing.Any]) -> None:
        """Set the current state of the environment."""
        self.scatter_pos = state['scatter_pos'].copy()
        self.step_count = state['step_count']

    def render(self):
        # Visualize current configuration
        actual_positions = []
        for i in range(0, 2 * self.n_scatterers, 2):
            x = self.scatter_pos[i] * (self.sx_scatterer/2)
            y = self.scatter_pos[i+1] * (self.sy_scatterer/2)
            actual_positions.append((x, y))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Draw cavity bounds
        ax.add_patch(Rectangle((-self.sx/2, -self.sy/2), self.sx, self.sy, 
                               fill=False, edgecolor='black', linewidth=2))
        
        # Draw waveguides (simplified)
        waveguide_positions = [
            {"x": -self.sx/2-self.waveguide_length/2, "y": self.waveguide_offset},
            {"x": -self.sx/2-self.waveguide_length/2, "y": -self.waveguide_offset},
            {"x": self.sx/2+self.waveguide_length/2, "y": self.waveguide_offset},
            {"x": self.sx/2+self.waveguide_length/2, "y": -self.waveguide_offset}
        ]
        
        for wg in waveguide_positions:
            ax.add_patch(Rectangle(
                (wg["x"]-self.waveguide_length/2, wg["y"]-self.waveguide_width/2),
                self.waveguide_length, self.waveguide_width,
                fill=False, edgecolor='blue', linewidth=1.5
            ))
        
        # Draw scatterers
        for i, pos in enumerate(actual_positions):
            ax.add_patch(Circle(pos, self.scatterer_radius, 
                               fill=True, alpha=0.7, facecolor='red'))
            ax.text(pos[0], pos[1], f"{i+1}", ha='center', va='center', 
                   color='white', fontweight='bold')
        
        ax.set_xlim(-self.sx/2-self.waveguide_length-1, self.sx/2+self.waveguide_length+1)
        ax.set_ylim(-self.sy/2-1, self.sy/2+1)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(f'Current Scatterer Configuration (Step {self.step_count})')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    env = BilliardTwoEnv()
    # # env = gym.make('MyEnv-v0')
    env.reset(seed=55)
    # print("check env begin")
    # check_env(env)
    # print("check env end")

    # print(env._calculate_tm(env.scatter_pos))

    tm_sample = env._calculate_tm(env.scatter_pos)
    print(env._calculate_reward(tm_sample))
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
