import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import meep as mp
import typing
from matplotlib.patches import Rectangle, Circle

from gymnasium import spaces

class BilliardBaseEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Common initialization parameters
        self.max_step = 2048   # for each episode, max steps we allowed
        self.n_scatterers = 20

        # MEEP simulation parameters
        self.resolution = 15  # pixels/cm
        self.n_runs = 100     # number of runs during simulation


        '''
        a = 0.01  chosen characteristic length = 1cm
        c = 3e8   speed of light
        target_freq_GHz = 15.0
        fsrc = target_freq_GHz * 1e9 * a / c = target_freq_GHz / 30
        '''
        self.fsrc = 15.0 / 30  # 15.0 GHz

        self.sx = 20
        self.sy = 20
        self.scatterer_radius = 0.5

        self.sx_scatterer = self.sx - 2 * self.scatterer_radius  # range of scatterer center 
        self.sy_scatterer = self.sy - 2 * self.scatterer_radius

        self.waveguide_width = 1.2
        self.waveguide_offset = 6.0  # waveguide center distance to billiard horizontal midline
        self.metal_thickness = 0.2
        
        self.pml_thickness = 3.0
        self.source_pml_distance = 3.0   # distance between source/montor and PML 
        self.source_billiard_distance = 6.0

        self.waveguide_length = self.source_pml_distance + self.source_billiard_distance + self.pml_thickness

        self.epsilon_bg = 1.0
        self.epsilon_scatter = 2.1
        
        self.mode_num = 1
        
        # Define action and observation spaces
        # both the action & obs space = n_scatterers * n_dim
        self.action_space = spaces.Box(low=-1, high=1, shape=(2 * self.n_scatterers,), dtype=np.float32) 
        # normalized coordinates, we should normalize the position by (length - 2*scatterer_radius)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2 * self.n_scatterers,), dtype=np.float32) 

        # For tracking progress
        self.best_error = float('inf')
        self.best_positions = None
        self.step_count = 0
        
        # Child classes should define these
        self.source_ports = []
        self.output_ports = []
        self.reflection_ports = []
        
        # Meep use the 'last object wins' principle, ie, if multiple objects overlap, later objects in the list take precedence. 
        # Allow overlapping simplifies implementation and help explore more diverse configurations.
        # Initial scatterer positions, this is normalized position !!!
        self.scatter_pos = self._generate_initial_positions()

        # mp.EVEN_Z + mp.ODD_Y -> Ex, Ey, Hz !=0; Ez, Hx, Hy =0
        # mp.EVEN_Y + mp.ODD_Z -> Ex, Ey, Hz =0;  Ez, Hx, Hy !=0
        # mp.EVEN_Y -> all elements !=0
        self.eig_parity = mp.EVEN_Y

    def _generate_initial_positions(self, seed=None):
        # Generate and normalize random positions
        # Use self.np_random instead of np.random to ensure proper seeding
        # self.np_random is provided by gym.Env and is properly seeded during reset()
        return self.np_random.uniform(low=-1, high=1, size=(2*self.n_scatterers,)).astype(np.float32)
    
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
    
    def _measure_incoming_amplitudes(self):
        """
        Measure the incoming field amplitude for a single source port.
        Since all source ports have identical waveguides, one measurement is sufficient.
        
        Returns:
            Incoming field amplitude
        """
        # Use the first source port as reference
        input_port = self.source_ports[0]
        
        # Create a simulation with just a single waveguide
        waveguide_only_geometry = []
        
        # Add only one waveguide
        self._create_metal_waveguide(
            waveguide_only_geometry,
            input_port["position"].x,
            input_port["position"].y,
            self.waveguide_length,
            self.waveguide_width
        )
        
        # Create simulation for reference incoming field
        cell_size = mp.Vector3(self.sx + 2*self.waveguide_length, self.sy + 2*self.metal_thickness)
        pml_layers = [mp.PML(self.pml_thickness, direction=mp.X)]
        
        sources = [mp.EigenModeSource(
            mp.ContinuousSource(frequency=self.fsrc),
            center=input_port["position"],
            size=mp.Vector3(0, self.waveguide_width-0.1),
            eig_band=self.mode_num,
            eig_parity=self.eig_parity
        )]
        
        ref_sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=waveguide_only_geometry,
            sources=sources,
            resolution=self.resolution,
            dimensions=2
        )
        
        # Add a monitor slightly in front of the source to measure incoming field
        # Place it a small distance in the direction of propagation
        monitor_pos = mp.Vector3(
            input_port["position"].x + 3,  # Slightly in front of source
            input_port["position"].y,
            input_port["position"].z
        )
        
        source_monitor = ref_sim.add_mode_monitor(
            self.fsrc, 0, 1,
            mp.ModeRegion(
                center=monitor_pos,
                size=mp.Vector3(0, self.waveguide_width-0.1)
            )
        )
        
        # Run reference simulation
        ref_sim.run(until=self.n_runs)  # Shorter run time is sufficient for straight waveguide
        
        # Get the incoming field amplitude
        source_data = ref_sim.get_eigenmode_coefficients(
            source_monitor, [1],
            eig_parity=self.eig_parity
        )
        incoming_amplitude = abs(source_data.alpha[0,0,0])
        
        # Clean up
        ref_sim.reset_meep()
        
        # print(incoming_amplitude)
        return incoming_amplitude

    def calculate_normalized_subSM(self, normalized_scatterers_positions, matrix_type="TM", visualize=False):
        """
        Calculate normalized scattering matrix (TM or RM) by dividing by the incoming field amplitude.
        
        Args:
            normalized_scatterers_positions: Normalized positions of scatterers
            matrix_type: "TM" for transmission matrix or "RM" for reflection matrix
            visualize: Boolean to enable visualization
            
        Returns:
            Normalized scattering matrix
        """
        # First, calculate the incoming field amplitudes at each source port
        incoming_amplitudes = self._measure_incoming_amplitudes()
        
        # Calculate the scattering matrix using the existing method
        sub_matrix = self._calculate_subSM(normalized_scatterers_positions, matrix_type, visualize)
        
        # Normalize the scattering matrix by dividing each row by the corresponding incoming amplitude
        normalized_matrix = sub_matrix / incoming_amplitudes
        
        return normalized_matrix

    def _calculate_subSM(self, normalized_scatterers_positions, matrix_type="TM", visualize=False):
        
        # Create geometry with scatterers
        geometry = self._create_full_geometry(normalized_scatterers_positions)
        
        # Calculate transmission matrix
        sub_matrix = []
        for input_port in self.source_ports:
            s_params = self._run_simulation_for_port(input_port, geometry, matrix_type, visualize)
            sub_matrix.append(s_params)
        
        return np.array(sub_matrix)
    
    def _run_simulation_for_port(self, input_port, geometry, matrix_type="TM", visualize=False):
        """Run simulation for a specific input port"""
        # Create a new simulation for this port
        cell_size = mp.Vector3(self.sx + 2*self.waveguide_length, self.sy + 2*self.metal_thickness)
        pml_layers = [mp.PML(self.pml_thickness, direction=mp.X)]
        
        sources = [mp.EigenModeSource(
            mp.ContinuousSource(frequency=self.fsrc),
            center=input_port["position"],
            size=mp.Vector3(0, self.waveguide_width-0.1),
            eig_band=self.mode_num,
            eig_parity=self.eig_parity
        )]
        
        sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=sources,
            resolution=self.resolution,
            dimensions=2
        )
        
        results = []

        if matrix_type == "TM":
            # Add monitors for all output ports
            mode_monitors = []
            for port in self.output_ports:
                monitor = sim.add_mode_monitor(
                    self.fsrc, 0, 1,
                    mp.ModeRegion(
                        center=port["position"],
                        size=mp.Vector3(0, self.waveguide_width-0.1)
                    )
                )
                mode_monitors.append(monitor)
            
            # sim.plot2D(plot_eps_flag=True)

            # Run simulation
            sim.run(until=self.n_runs)
            
            if visualize:
                plt.figure()
                field_func = lambda x: np.sqrt(np.abs(x)) # lambda x: 20*np.log10(np.abs(x))
                sim.plot2D(fields=mp.Ez,
                        field_parameters={'alpha':1, 'cmap':'hsv', 'interpolation':'spline36', 'post_process':field_func, 'colorbar':False},
                        boundary_parameters={'hatch':'o', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.3},
                        eps_parameters={'alpha':1, 'contour':False}
                    )
                # plt.xlim(-self.sx/2 - 5, self.sx/2 + 5)
                plt.show()

            # Calculate transmission coefficients for all output ports      
            for monitor in mode_monitors:
                mode_data = sim.get_eigenmode_coefficients(
                    monitor, [1],
                    eig_parity=self.eig_parity
                )
                results.append(mode_data.alpha[0,0,0])

        else:
            # Add monitors for all reflection ports
            mode_monitors = []
            for port in self.reflection_ports:
                monitor = sim.add_mode_monitor(
                    self.fsrc, 0, 1,
                    mp.ModeRegion(
                        center=port["position"],
                        size=mp.Vector3(0, self.waveguide_width-0.1)
                    )
                )
                mode_monitors.append(monitor)

            # Run simulation
            sim.run(until=self.n_runs)
            
            if visualize:
                plt.figure()
                field_func = lambda x: np.sqrt(np.abs(x)) # lambda x: 20*np.log10(np.abs(x))
                sim.plot2D(fields=mp.Ez,
                        field_parameters={'alpha':1, 'cmap':'hsv', 'interpolation':'spline36', 'post_process':field_func, 'colorbar':False},
                        boundary_parameters={'hatch':'o', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.3},
                        eps_parameters={'alpha':1, 'contour':False}
                    )
                # plt.xlim(-self.sx/2 - 5, self.sx/2 + 5)
                plt.show()

            # Calculate transmission coefficients for all output ports
            for monitor in mode_monitors:
                mode_data = sim.get_eigenmode_coefficients(
                    monitor, [1],
                    eig_parity=self.eig_parity
                )
                results.append(mode_data.alpha[0,0,1])

        sim.reset_meep()  # Explicitly reset MEEP to free memory
        return results
        
    def step(self, action) -> tuple[spaces.Box, np.float32, bool, bool, dict[str, typing.Any]]:
        self.step_count += 1

        # Apply action (small adjustments to positions)
        scaling_factor = 0.001  # Control adjustment size
        self.scatter_pos = np.clip(self.scatter_pos + action * scaling_factor, -1, 1)
        
        # Calculate sub SM with new positions
        subSM = self._calculate_subSM(self.scatter_pos, matrix_type="TM", visualize=False)
        
        # Calculate reward and error
        reward, error = self._calculate_reward(subSM)
        
        # Update best positions if current error is lower than best error
        if error < self.best_error:
            self.best_error = error
            self.best_positions = self.scatter_pos.copy()  # Make a copy to prevent reference issues

        # Check if goal is achieved or max steps reached
        terminated = error < 0.01         #  5% deviation for 1.73*t11 vs t21, 5% deviation for 1.73*t12 vs t22, error_threshold = 2 * (5%)^2 = 0.005
        
        truncated = self.step_count >= self.max_step

        info = {
            "error": error,
            "scatter_pos": self.scatter_pos.copy(),
            "step": self.step_count
        }
        
        # Print progress every 10 steps
        # if self.step_count % 10 == 0:
        #     print(f"Step {self.step_count}, Error: {error:.6f}, Reward: {reward:.6f}")

        return self.scatter_pos, reward, terminated, truncated, info

    def reset(self, seed=None, options=None) -> tuple[spaces.Box, dict[str, typing.Any]]:
        """
        At the beginning of each episode, reset env.

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


        self.step_count = 0

        # Return both observation and info dict
        return self.scatter_pos, {}
    
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

    def plot_field_intensity(self, sim, component=mp.Ez):
        output_plane=mp.Volume(center=mp.Vector3(), 
                               size=mp.Vector3(self.sx + 2*self.waveguide_length, self.sy + 2*self.metal_thickness))
        
        # Get the field data
        field_data = sim.get_array(center=output_plane.center, size=output_plane.size, component=component)
        print(np.shape(field_data))
        # Calculate intensity (|E|²)
        intensity = np.abs(field_data)**2
    
        # Plot the intensity
        plt.figure()
        # plt.imshow(intensity.transpose(), interpolation='spline36', cmap='magma') 
        plt.imshow(intensity.transpose())
        plt.colorbar(label='Intensity')
        plt.title(f'{component} Intensity')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.tight_layout()
        # plt.savefig(f"{component}_intensity.png")
        plt.show()

    def plot_field_cross_section(self, sim, start_point, end_point, component=mp.Ex,
                             num_points=100, plot_abs=True):
        """
        Plot field values along a line between two points.

        Args:
            sim: The MEEP simulation
            start_point: Vector3 starting coordinate
            end_point: Vector3 ending coordinate
            component: Field component to plot
            num_points: Number of points to sample along the line
            plot_abs: Whether to plot absolute value or real part
        """
        # Create interpolation points along the line
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            x = start_point.x + t * (end_point.x - start_point.x)
            y = start_point.y + t * (end_point.y - start_point.y)
            z = start_point.z + t * (end_point.z - start_point.z)
            points.append(mp.Vector3(x, y, z))

        # Get field values at each point
        field_values = [sim.get_field_point(component, p) for p in points]

        # Calculate distances along the path for x-axis
        distances = [0]
        for i in range(1, len(points)):
            p1 = points[i - 1]
            p2 = points[i]
            dist = np.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p1.z) ** 2)
            distances.append(distances[-1] + dist)

        # Plot the field values
        plt.figure(figsize=(10, 6))
        if plot_abs:
            plt.plot(distances, [abs(f) for f in field_values], 'b-', linewidth=2)
            plt.plot(distances, [abs(f) ** 2 for f in field_values], 'r--', linewidth=2)
            plt.legend(['|Field|', '|Field|²'])
            plt.ylabel(f'|{component}| Amplitude')
        else:
            plt.plot(distances, [f.real for f in field_values], 'b-', linewidth=2)
            plt.plot(distances, [f.imag for f in field_values], 'r--', linewidth=2)
            plt.legend(['Re(Field)', 'Im(Field)'])
            plt.ylabel(f'{component} Field Value')

        plt.xlabel('Distance along path')
        plt.title(f'{component} Field Along Cross-Section')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def visualize_selective_power_flow(self, sim):
        """
        Visualize power flow with arrows only in areas of significant power flow
        """
        # Get field components
        ex = sim.get_array(component=mp.Ex)
        ey = sim.get_array(component=mp.Ey)
        hz = sim.get_array(component=mp.Hz)
        
        # For 2D TE mode (Ex, Ey, Hz), Poynting vector components are:
        sx = ey * np.conj(hz)  # x-component
        sy = -ex * np.conj(hz) # y-component
        
        # Magnitude of power flow
        power_magnitude = np.sqrt(np.sqrt(np.abs(sx)**2 + np.abs(sy)**2))
        
        # Plot power flow magnitude
        plt.figure()
        plt.imshow(power_magnitude.transpose(), origin='lower', cmap='viridis')
        plt.colorbar(label='Power flow magnitude')
        plt.title('Power Flow Distribution')
        
        # Add power flow direction vectors ONLY IN SIGNIFICANT AREAS
        step = 12  # Spacing between arrows
        
        # Create grid
        x = np.arange(0, ex.shape[0], step)
        y = np.arange(0, ex.shape[1], step)
        X, Y = np.meshgrid(x, y)
        
        # Get downsampled data
        sx_ds = sx[::step, ::step]
        sy_ds = sy[::step, ::step]
        power_ds = power_magnitude[::step, ::step]
        
        # Threshold for significant power flow (adjust as needed)
        # Find the value at 30% of the max power
        threshold = 0.3 * np.max(power_magnitude)
        
        # Create masks for points to plot
        mask = power_ds > threshold
        
        # Only plot points above threshold
        X_plot = X[mask.T]
        Y_plot = Y[mask.T]
        
        # Normalize direction vectors 
        sx_plot = np.real(sx_ds[mask])
        sy_plot = np.real(sy_ds[mask])
        norm = np.sqrt(sx_plot**2 + sy_plot**2)
        sx_plot = sx_plot / (norm + 1e-10)
        sy_plot = sy_plot / (norm + 1e-10)
        
        # Plot selective quiver
        plt.quiver(X_plot, Y_plot, sx_plot, sy_plot, 
                scale=15,      
                pivot='mid',    
                color='white', 
                alpha=0.9,
                width=0.006,
                headwidth=5,
                headlength=6)
        
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        plt.show()

    def render(self):
        """Render the current state of the environment"""
        # Convert normalized positions to actual coordinates
        actual_positions = []
        for i in range(0, 2 * self.n_scatterers, 2):
            x = self.scatter_pos[i] * (self.sx_scatterer/2)
            y = self.scatter_pos[i+1] * (self.sy_scatterer/2)
            actual_positions.append((x, y))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Draw cavity bounds
        ax.add_patch(Rectangle((-self.sx/2, -self.sy/2), self.sx, self.sy, 
                            fill=False, edgecolor='black', linewidth=2))
        
        # Draw waveguides based on port definitions
        for port in self.source_ports + self.output_ports:
            # Get waveguide position
            x_center = port["position"].x
            y_center = port["position"].y
            
            # Draw waveguide outline
            ax.add_patch(Rectangle(
                (x_center-self.waveguide_length/2, y_center-self.waveguide_width/2),
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

    def _create_base_geometry(self):
        """
        Create the base geometry (billiard and waveguides).
        Child classes should implement this method.
        """
        raise NotImplementedError("Subclasses must implement _create_base_geometry")
    
    def _calculate_reward(self, tm):
        """
        Calculate reward based on the transmission matrix.
        Child classes should implement this method.
        """
        raise NotImplementedError("Subclasses must implement _calculate_reward")
