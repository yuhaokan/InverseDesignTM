import numpy as np
import meep as mp
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class BilliardTwoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.n_scatterers = 3
        # Define action and observation spaces

        # both the action & obs space = n_scatterers * n_dim
        self.action_space = spaces.Box(low=-1, high=1, shape=(2 * self.n_scatterers,), dtype=np.float32) 

        # normalized coordinates, we should normalize the position by (length - 2*scatterer_radius)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2 * self.n_scatterers,), dtype=np.float32) 
        
        # MEEP simulation parameters
        self.resolution = 25  # pixels/cm

        '''
        a = 0.01  chosen characteristic length = 1cm
        c = 3e8   speed of light
        target_freq_GHz = 15.0
        fsrc = target_freq_GHz * 1e9 * a / c = target_freq_GHz / 30
        '''
        self.fsrc = 15.0 / 30

        self.sx = 20
        self.sy = 10
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
    
    def _generate_initial_positions(self, seed=42):
        # Generate and normalize random positions
        np.random.seed(seed)
        return np.random.uniform(low=-1, high=1, size=(2*self.n_scatterers,))

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
    
    def _calculate_reward(self, tm):
        # enforce the non-overlapping rule in step function, do not modify reward to penalize scatterer ovelapping

        # Target relationship: tm[0] * 1.73 = tm[1], expect a rank-1 TM
        error = np.sum(np.abs(tm[0] * 1.73 - tm[1])**2) 
        
        # Reward is negative of error (higher reward for lower error)
        reward = -error
        
        return reward, error
    
    def _modify_action_with_constraints(self, action):
        """Modify the action to prevent overlapping"""
        current_pos = self.scatter_pos.reshape(-1, 2)
        action = action.reshape(-1, 2)
        modified_action = action.copy()
    
        # Predict new positions
        new_pos = current_pos + action
    
        min_distance = 2 * self.scatterer_radius
        
        # Check and adjust each pair of scatterers
        for i in range(len(new_pos)):
            for j in range(i + 1, len(new_pos)):
                diff = new_pos[i] - new_pos[j]
                distance = np.linalg.norm(diff)
            
                if distance < min_distance:
                    # Calculate the overlap
                    overlap = min_distance - distance
                    direction = diff / distance
                
                    # Move scatterers apart
                    modified_action[i] += 0.5 * overlap * direction
                    modified_action[j] -= 0.5 * overlap * direction
    
        return modified_action.reshape(-1)

    def step(self, action):
        self.step_count += 1

        # Apply action (small adjustments to positions)
        scaling_factor = 0.05  # Control adjustment size

        # Modify action to prevent overlapping
        modified_action = self._modify_action_with_constraints(action * scaling_factor)

        self.scatter_pos = np.clip(self.scatter_pos + modified_action, -1, 1)
        
        # Calculate transmission matrix with new positions
        tm = self._calculate_tm(self.scatter_pos)
        
        # Calculate reward and error
        reward, error = self._calculate_reward(tm)
        
        # Check if goal is achieved or max steps reached
        done = (error < 0.01) or (self.step_count >= 1000)           # here we set the max step to be 1000?
        
        info = {
            "error": error,
            "tm_0": tm[0],
            "tm_1": tm[1],
            "target": tm[0] * 1.73 - tm[1],
            "step": self.step_count
        }
        
        # Print progress every 10 steps
        if self.step_count % 10 == 0:
            print(f"Step {self.step_count}, Error: {error:.6f}, Reward: {reward:.6f}")
            
        return self.scatter_pos, reward, done, info
    
    # def reset(self, *, seed=None):
    #     if seed is not None:
    #         np.random.seed(seed)

    #     # Optionally reset to best known positions with small perturbation
    #     if self.best_positions is not None and np.random.random() < 0.7:
    #         # 70% chance to use best positions with noise
    #         noise = np.random.normal(0, 0.05, size=6)  # Small Gaussian noise
    #         self.scatter_pos = np.clip(self.best_positions + noise, -1, 1)
    #     else:
    #         # 30% chance to generate new random positions
    #         self.scatter_pos = self._generate_initial_positions()
        
    #     self.step_count = 0
    #     return self.scatter_pos

    # at the beginning of each episode, reset env
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional options for reset

        Returns:
            tuple: (observation, info_dict)
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Optionally reset to best known positions with small perturbation
        if self.best_positions is not None and np.random.random() < 0.7:
            # 70% chance to use best positions with noise
            noise = np.random.normal(0, 0.05, size=6)  # Small Gaussian noise
            self.scatter_pos = np.clip(self.best_positions + noise, -1, 1)
        else:
            # 30% chance to generate new random positions
            self.scatter_pos = self._generate_initial_positions(seed=seed)

        self.step_count = 0

        # Get observation
        observation = self._get_observation()

        # Create info dictionary
        info = {
            "scatter_positions": self.scatter_pos.copy(),
            "reset_seed": seed
        }

        # Return both observation and info dict
        return observation, info

    def _get_observation(self):
        # Make sure observation matches the defined space
        observation = np.array(..., dtype=np.float32)
        assert self.observation_space.contains(observation), "Invalid observation!"
        return observation
    
    def render(self, mode='human'):
        # Visualize current configuration
        actual_positions = []
        for i in range(0, 6, 2):
            x = self.scatter_pos[i] * (self.sx/2)
            y = self.scatter_pos[i+1] * (self.sy/2)
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
    print(env._calculate_tm(env.scatter_pos))