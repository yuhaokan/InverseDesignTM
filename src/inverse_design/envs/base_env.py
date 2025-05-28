import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import meep as mp
import typing
from matplotlib.patches import Rectangle, Circle

from gymnasium import spaces

class BilliardBaseEnv(gym.Env):
    def __init__(self, target_type="Rank1"):
        super().__init__()

        self.target_type = target_type

        # Common initialization parameters
        self.max_step = 1024   # for each episode, max steps we allowed
        self.terminated_threshold = 0.01
        self.n_scatterers = 20

        # MEEP simulation parameters
        self.resolution = 15  # pixels/cm

        # env4, 5, 6, 11 -> 100
        # env10, 12, 20 -> 200
        self.n_runs = 200     # number of runs during simulation ############################################################### hyper-parameter 1


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
        # else 6.0
        # env20 -> 9.0
        self.waveguide_offset = 6.0  # waveguide center distance to billiard horizontal midline ################################# hyper-parameter 5
        self.metal_thickness = 0.2

        # env4, 5, 6 -> 0.1
        # env10, 11, 12, 20  -> 0.2
        self.source_length_diff = 0.2  # diff between source & waveguide_width    ####################################################################  hyper-parameter 2
        
        self.pml_thickness = 3.0
        self.source_pml_distance = 3.0   # distance between source/montor and PML 
        self.source_billiard_distance = 6.0

        self.waveguide_length = self.source_pml_distance + self.source_billiard_distance + self.pml_thickness

        self.epsilon_bg = 1.0

        # env4 -> 3.9
        # env5,6,10, 11,12 -> 2.1
        # env20 -> 10
        self.epsilon_scatter = 2.1     ####################################################################  hyper-parameter 3
        
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

        # env4, 5 -> mp.EVEN_Z + mp.ODD_Y
        # env6, 11, 12, 20 -> mp.EVEN_Y + mp.ODD_Z
        # env10 -> mp.NO_PARITY
        # mp.EVEN_Z + mp.ODD_Y -> Ex, Ey, Hz !=0; Ez, Hx, Hy =0
        # mp.EVEN_Y + mp.ODD_Z -> Ex, Ey, Hz =0;  Ez, Hx, Hy !=0
        # mp.EVEN_Y -> all elements !=0
        # mp.NO_PARITY
        self.eig_parity = mp.EVEN_Y + mp.ODD_Z                   ####################################################### hyper-parameter 4

        self.field_func = lambda x: (np.abs(x))

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
        
        # Get loss/gain factor, default to 0 if not set
        loss_factor = getattr(self, 'uniform_loss_factor', 0)

        if loss_factor != 0:
            # Angular frequency: ω = 2πf
            omega = 2 * np.pi * self.fsrc

            # Calculate conductivity: positive for loss, negative for gain
            # σ = ω*ε₀*loss_factor
            conductivity = omega * loss_factor

            # For stability in gain media, use a saturable gain model with high saturation
            if loss_factor < 0:  # Gain medium
                # Create gain medium
                gain_medium = mp.Medium(
                    epsilon=self.epsilon_bg,
                    E_susceptibilities=[
                        # Negative gamma provides gain
                        mp.LorentzianSusceptibility(
                            sigma=-abs(loss_factor) * omega,  # Gain strength
                            frequency=self.fsrc,  # Center frequency
                            gamma=-1e-5  # Small negative value for gain
                        )
                    ]
                )

                geometry.append(mp.Block(
                    material=gain_medium,
                    center=mp.Vector3(0, 0),
                    size=mp.Vector3(self.sx, self.sy)
                ))
            else:  # Loss medium
                # Create lossy medium using D_conductivity
                lossy_medium = mp.Medium(
                    epsilon=self.epsilon_bg,
                    D_conductivity=conductivity  # Positive for loss
                )

                geometry.append(mp.Block(
                    material=lossy_medium,
                    center=mp.Vector3(0, 0),
                    size=mp.Vector3(self.sx, self.sy)
                ))
                
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
            size=mp.Vector3(0, self.waveguide_width - self.source_length_diff),
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
                size=mp.Vector3(0, self.waveguide_width - self.source_length_diff)
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
        
        return incoming_amplitude  # 73.7

    def _calculate_normalized_subSM(self, normalized_scatterers_positions, matrix_type="TM", visualize=False):
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
    
    def _run_simulation_for_port(self, input_port, geometry, matrix_type="TM", visualize=False, field_component=mp.Ez):
        """Run simulation for a specific input port"""
        # Create a new simulation for this port
        cell_size = mp.Vector3(self.sx + 2*self.waveguide_length, self.sy + 2*self.metal_thickness)
        pml_layers = [mp.PML(self.pml_thickness, direction=mp.X)]
        
        sources = [mp.EigenModeSource(
            mp.ContinuousSource(frequency=self.fsrc),
            center=input_port["position"],
            size=mp.Vector3(0, self.waveguide_width - self.source_length_diff),
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
                        size=mp.Vector3(0, self.waveguide_width - self.source_length_diff)
                    )
                )
                mode_monitors.append(monitor)
            
            # sim.plot2D(plot_eps_flag=True)

            # Run simulation
            sim.run(until=self.n_runs)
            
            if visualize:
                plt.figure()
                sim.plot2D(fields=field_component,
                        field_parameters={'alpha':1, 'cmap': 'viridis', 'interpolation':'spline36', 'post_process':self.field_func, 'colorbar':True},  # 'cmap':'hsv'
                        boundary_parameters={'hatch':'o', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.3},
                        eps_parameters={'alpha':0.8, 'contour':False}
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
                        size=mp.Vector3(0, self.waveguide_width - self.source_length_diff)
                    )
                )
                mode_monitors.append(monitor)

            # Run simulation
            sim.run(until=self.n_runs)
            
            if visualize:
                plt.figure()
                sim.plot2D(fields=field_component,
                        field_parameters={'alpha':1, 'cmap':'viridis', 'interpolation':'spline36', 'post_process':self.field_func, 'colorbar':True},   # 'cmap':'hsv'
                        boundary_parameters={'hatch':'o', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.3},
                        eps_parameters={'alpha':0.8, 'contour':False}
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
        scaling_factor = 0.005  # Control adjustment size
        self.scatter_pos = np.clip(self.scatter_pos + action * scaling_factor, -1, 1)
        
        # Calculate sub SM with new positions
        subSM = self._calculate_subSM(self.scatter_pos, matrix_type="TM", visualize=False)
        
        # Calculate reward and error
        reward, error = self._calculate_reward(tm=subSM, target_type=self.target_type)
        
        # Update best positions if current error is lower than best error
        if error < self.best_error:
            self.best_error = error
            self.best_positions = self.scatter_pos.copy()  # Make a copy to prevent reference issues

        # Check if goal is achieved or max steps reached
        terminated = error < self.terminated_threshold         #  5% deviation for 1.73*t11 vs t21, 5% deviation for 1.73*t12 vs t22, error_threshold = 2 * (5%)^2 = 0.005
        
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
            noise = self.np_random.normal(0, 0.005, size=(2*self.n_scatterers,)).astype(np.float32)
            self.scatter_pos = np.clip(self.best_positions + noise, -1, 1)
        else:
            # 30% chance to generate new random positions
            self.scatter_pos = self._generate_initial_positions(seed)


        self.step_count = 0

        # Return both observation and info dict
        return self.scatter_pos, {}

    def plot_field_intensity(self, sim, component=mp.Ez):
        output_plane=mp.Volume(center=mp.Vector3(), 
                               size=mp.Vector3(self.sx + 2*self.waveguide_length, self.sy + 2*self.metal_thickness))
        
        # Get the field data
        field_data = sim.get_array(center=output_plane.center, size=output_plane.size, component=component)
        intensity = np.abs(field_data)**2
    
        plt.figure()
        plt.imshow(intensity.transpose())
        plt.colorbar(label='Intensity')
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

    def plot_lowest_transmission_eigenchannel(self, scatter_positions, field_component=mp.Ez):
        """
        Plot the field pattern of the lowest transmission eigenchannel.
        
        Args:
            scatter_positions: Normalized positions of the scatterers
        """
        # Calculate the transmission matrix
        tm = self._calculate_normalized_subSM(scatter_positions, matrix_type="TM")
        
        tm = tm.T   # convert to standard notation

        # Perform SVD on the transmission matrix
        U, S, Vh = np.linalg.svd(tm)
        
        # The lowest transmission eigenchannel corresponds to the smallest singular value
        min_idx = np.argmin(S)

        # Overwrite to plot the higheset eigenchannel
        # min_idx = 0

        # Get the input state (right singular vector) corresponding to the lowest eigenchannel
        v_min = Vh[min_idx, :].conj()  # Complex conjugate for correct phase
        
        print(f"Singular values: {S}")
        print(f"Lowest transmission channel (singular value = {S[min_idx]}) selected")
        
        # Create geometry with scatterers
        geometry = self._create_full_geometry(scatter_positions)
        
        # Create a custom source that excites the eigenchannel
        sources = []
        for i, port in enumerate(self.source_ports):
            # Use the eigenvector component as amplitude/phase for each port
            amplitude = v_min[i]
            sources.append(mp.EigenModeSource(
                mp.ContinuousSource(frequency=self.fsrc),
                center=port["position"],
                size=mp.Vector3(0, self.waveguide_width - self.source_length_diff),
                eig_band=self.mode_num,
                eig_parity=self.eig_parity,
                amplitude=amplitude
            ))
        
        # Setup simulation
        cell_size = mp.Vector3(self.sx + 2*self.waveguide_length, self.sy + 2*self.metal_thickness)
        pml_layers = [mp.PML(self.pml_thickness, direction=mp.X)]
        
        sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=sources,
            resolution=self.resolution,
            dimensions=2
        )
        
        # Run simulation
        sim.run(until=self.n_runs)
        
        # Plot the field
        plt.figure(figsize=(12, 10))
        
        # Plot field intensity
        sim.plot2D(fields=field_component,
                field_parameters={'alpha': 1, 'cmap': 'viridis', 
                                'interpolation': 'spline36', 
                                'post_process': self.field_func, 
                                'colorbar': True},
                boundary_parameters={'hatch': 'o', 'linewidth': 1.5, 
                                    'facecolor': 'none', 'edgecolor': 'k', 
                                    'alpha': 0.3},
                eps_parameters={'alpha': 0.8, 'contour': False}
                )
        
        plt.title(f"Field Pattern of Lowest Transmission Eigenchannel (σ = {S[min_idx]:.4f})")
        plt.tight_layout()
        plt.show()
        
        # Clean up
        sim.reset_meep()

    def plot_lowest_transmission_eigenchannel_steady_state(self, scatter_positions=None, field_component=mp.Ez, matrix_type="TM"):
        """
        Plot the steady-state field pattern of the lowest eigenchannel (highest transmission/lowest loss).
        
        Args:
            scatter_positions: Positions of scatterers (uses current positions if None)
            field_component: Field component to plot (default: mp.Ez)
            n_modes: Number of modes to compute when finding the lowest eigenchannel
        """
        dft_resolution = 1

        if scatter_positions is None:
            scatter_positions = self.scatter_pos
        
        # Create geometry with scatterers
        geometry = self._create_full_geometry(scatter_positions)
        
        # Setup simulation cell
        cell_size = mp.Vector3(self.sx + 2 * self.waveguide_length, self.sy + 2 * self.metal_thickness)
        pml_layers = [mp.PML(self.pml_thickness, direction=mp.X)]
        
        # First, calculate the full scattering matrix
        # We need all source ports and output ports to construct the matrix
        tm = self._calculate_normalized_subSM(scatter_positions, matrix_type, visualize=False)
        tm = tm.T

        # Perform Singular Value Decomposition (SVD) to find eigenchannels

        U, S, Vh = np.linalg.svd(tm)
        print(f"Singular values: {S}")
        
        min_idx = np.argmin(S)
        print(f"Lowest Singular values: {S[min_idx]}")
        # The highest singular value corresponds to the lowest-loss eigenchannel
        # The corresponding right singular vector tells us how to excite this channel
        eigenchannel_weights = Vh[min_idx].conj()  # row of V† matrix
            
        # Normalize the weights
        eigenchannel_weights = eigenchannel_weights / np.linalg.norm(eigenchannel_weights)
        
        # Create a simulation with sources using the eigenchannel weights
        sources = []
        for i, input_port in enumerate(self.source_ports):
            # Create a source with amplitude and phase from eigenchannel weights
            weight = eigenchannel_weights[i]
            # amplitude = np.abs(weight)
            # phase = np.angle(weight)
            
            # Create source with proper amplitude and phase
            source = mp.EigenModeSource(
                mp.ContinuousSource(frequency=self.fsrc),
                center=input_port["position"],
                size=mp.Vector3(0, self.waveguide_width - self.source_length_diff),
                eig_band=self.mode_num,
                eig_parity=self.eig_parity,
                amplitude=weight
            )
            sources.append(source)
        
        # Create simulation with these weighted sources
        sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=sources,
            resolution=self.resolution,
            dimensions=2
        )
        
        # Add DFT monitor for the entire cell
        dft = sim.add_dft_fields([field_component], 
                                self.fsrc, self.fsrc, 1,
                                center=mp.Vector3(),
                                size=cell_size,
                                resolution=dft_resolution)
        
        # Run simulation until steady state
        sim.run(until=self.n_runs)
        
        # Get the DFT field data
        dft_data = sim.get_dft_array(dft, field_component, 0)
        
        # Get the grid dimensions for plotting
        nx, ny = dft_data.shape
        x = np.linspace(-cell_size.x/2, cell_size.x/2, nx)
        y = np.linspace(-cell_size.y/2, cell_size.y/2, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Calculate intensity
        plot_data = self.field_func(dft_data)
        
        # Create figure and plot
        plt.figure(figsize=(12, 10))
        
        # Plot intensity
        plt.imshow(plot_data.T, 
                origin='lower', 
                extent=[-cell_size.x/2, cell_size.x/2, -cell_size.y/2, cell_size.y/2],
                cmap='viridis', 
                interpolation='bilinear')

        plt.colorbar(label='Log Intensity log(|E|²)')
        plt.tight_layout()
        plt.show()

        # Clean up
        sim.reset_meep()

    
    def plot_speckle_patterns(self, scatter_positions=None, field_component=mp.Ez):
        """
        Plot the speckle pattern (field distribution) for each input port excited individually
        using sim.plot2D() for visualization.
        
        Args:
            scatter_positions: Normalized positions of scatterers
            field_component: Which field component to visualize (default: mp.Ez)
        """

        if scatter_positions is None:
            scatter_positions = self.scatter_pos

        # Create geometry with scatterers
        geometry = self._create_full_geometry(scatter_positions)
        
        # Set up figure with subplots for each input port
        n_ports = len(self.source_ports)
        fig, axes = plt.subplots(n_ports, 1, figsize=(8, 6*n_ports))
        if n_ports == 1:  # Handle case with a single input port
            axes = [axes]
        
        for idx, input_port in enumerate(self.source_ports):
            # Create simulation for this port
            cell_size = mp.Vector3(self.sx + 2*self.waveguide_length, self.sy + 2*self.metal_thickness)
            pml_layers = [mp.PML(self.pml_thickness, direction=mp.X)]
            
            sources = [mp.EigenModeSource(
                mp.ContinuousSource(frequency=self.fsrc),
                center=input_port["position"],
                size=mp.Vector3(0, self.waveguide_width - self.source_length_diff),
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
            
            # Run simulation
            sim.run(until=self.n_runs)
            
            # Set the current axis for plot2D to use
            plt.sca(axes[idx])
            
            # Plot the field using plot2D
            sim.plot2D(
                fields=field_component,
                field_parameters={
                    'alpha': 1,
                    'cmap': 'viridis', 
                    'interpolation': 'spline36',
                    'post_process': self.field_func,
                    'colorbar': True
                },
                boundary_parameters={
                    'hatch': 'o', 
                    'linewidth': 1.5, 
                    'facecolor': 'none', 
                    'edgecolor': 'k', 
                    'alpha': 0.3
                },
                eps_parameters={
                    'alpha': 0.8, 
                    'contour': False
                }
            )
            
            axes[idx].set_title(f"Input Port: {input_port['name']}")
            
            # Clean up before next simulation
            sim.reset_meep()
        
        plt.tight_layout()
        plt.suptitle("Speckle Patterns for Individual Input Ports", fontsize=16, y=0.98)
        plt.show()
        
        return fig, axes

    def plot_speckle_patterns_steady_state(self, scatter_positions=None, field_component=mp.Ez, input_port_index=0):

        dft_resolution = 1

        """Plot steady-state fields at a single frequency using MEEP's output_dft"""
        if scatter_positions is None:
            scatter_positions = self.scatter_pos
        
        # Create geometry with scatterers
        geometry = self._create_full_geometry(scatter_positions)
        
        # Get the input port
        input_port = self.source_ports[input_port_index]
        
        # Setup simulation
        cell_size = mp.Vector3(self.sx + 2 * self.waveguide_length, self.sy + 2 * self.metal_thickness)
        pml_layers = [mp.PML(self.pml_thickness, direction=mp.X)]
        
        sources = [mp.EigenModeSource(
            mp.ContinuousSource(frequency=self.fsrc),
            center=input_port["position"],
            size=mp.Vector3(0, self.waveguide_width - self.source_length_diff),
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
        
        # Add DFT monitor for the entire cell
        dft = sim.add_dft_fields([field_component], 
                                self.fsrc, self.fsrc, 1,
                                center=mp.Vector3(),
                                size=cell_size,
                                resolution=dft_resolution)
        
        # Run simulation until steady state
        sim.run(until=self.n_runs)
        
        # Create a new figure
        plt.figure(figsize=(10, 8))
    

        dft_data = sim.get_dft_array(dft, field_component, 0)

        # Get the grid dimensions for the DFT data
        nx, ny = dft_data.shape
        x = np.linspace(-cell_size.x/2, cell_size.x/2, nx)
        y = np.linspace(-cell_size.y/2, cell_size.y/2, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Get the dielectric structure for overlay (can use lower resolution too)
        eps_data = sim.get_epsilon()
        

        # Plot intensity (magnitude squared)
        plot_data = self.field_func(dft_data) # np.log(np.abs(dft_data)**2 + 1e-6)
        
        plt.imshow(plot_data.T, 
                   origin='lower', 
                   extent=[-cell_size.x/2, cell_size.x/2, -cell_size.y/2, cell_size.y/2],
                   cmap='viridis', 
                   vmin=0, 
                   interpolation='bilinear')
        
        
        # Add colorbar and labels
        plt.colorbar(label='Intensity')
        plt.tight_layout()
        

        plt.show()

        sim.reset_meep()
        
        return plot_data
        
    def plot_phase_map(self, scatter_pos, freq_range=(0.45, 0.55), freq_points=3,
                       loss_range=(-0.05, 0.05), loss_points=3, save_path=None):
        """
        Plot a phase map of det(TM) angle vs frequency and loss factor.

        Args:
            scatter_pos: Position of scatterers (normalized)
            freq_range: Tuple of (min_freq, max_freq) around self.fsrc
            freq_points: Number of frequency points to sample
            loss_range: Tuple of (min_loss, max_loss) for uniform loss
            loss_points: Number of loss points to sample
            save_path: Path to save the figure (if None, display only)

        Returns:
            Fig, ax objects of the generated plot
        """

        # Create frequency and loss arrays
        freqs = np.linspace(freq_range[0], freq_range[1], freq_points)
        losses = np.linspace(loss_range[0], loss_range[1], loss_points)

        # Initialize results array for det(TM) angle
        det_angles = np.zeros((loss_points, freq_points), dtype=np.float64)
        det_magnitudes = np.zeros((loss_points, freq_points), dtype=np.float64)

        # Store original values to restore later
        original_freq = self.fsrc
        original_loss = getattr(self, 'uniform_loss_factor', 0)

        # Add uniform loss factor attribute if not present
        if not hasattr(self, 'uniform_loss_factor'):
            self.uniform_loss_factor = 0

        # Set up progress tracking
        total_iterations = loss_points * freq_points
        current_iteration = 0

        try:
            # Loop over loss values and frequencies
            for i, loss in enumerate(losses):
                self.uniform_loss_factor = loss

                for j, freq in enumerate(freqs):
                    # Update frequency
                    self.fsrc = freq

                    # Calculate TM with current settings
                    tm = self._calculate_subSM(scatter_pos, matrix_type="TM", visualize=False)

                    # Calculate determinant and its phase angle
                    det = tm[0][0] * tm[1][1] - tm[0][1] * tm[1][0]
                    angle = np.angle(det, deg=False)
                    magnitude = np.abs(det)

                    det_angles[i, j] = angle
                    det_magnitudes[i, j] = magnitude

                    # Update progress
                    current_iteration += 1
                    if current_iteration % 5 == 0 or current_iteration == total_iterations:
                        print(f"Progress: {current_iteration}/{total_iterations} iterations completed")

        finally:
            # Restore original values
            self.fsrc = original_freq
            self.uniform_loss_factor = original_loss

            
        # Save if path provided
        if save_path:
            np.savez(save_path + 'det_phase_map.npz', freqs=freqs, losses=losses, det_angles=det_angles, det_magnitudes=det_magnitudes)

        # Create figure with two subplots - phase and magnitude
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot phase map
        pcm1 = ax1.pcolormesh(
            freqs,
            -losses,   # put loss in lower half plane
            det_angles,
            cmap='hsv',
            vmin=-np.pi,
            vmax=np.pi,
            shading='auto'
        )

        # Add colorbar for phase
        cbar1 = fig.colorbar(pcm1, ax=ax1, label='Phase Angle of det(TM)')

        # Plot magnitude map (log scale for better visualization)
        pcm2 = ax2.pcolormesh(
            freqs,
            -losses,    # put loss in lower half plane
            np.log10(det_magnitudes),
            cmap='viridis',
            shading='auto'
        )

        # Add colorbar for magnitude
        cbar2 = fig.colorbar(pcm2, ax=ax2, label='Log10 Magnitude of det(TM)')

        # Add labels and titles
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Loss/Gain Factor (negative = loss)')
        ax1.set_title('Phase Map of Determinant')
        # ax1.grid(True, linestyle='--', alpha=0.7)

        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Loss/Gain Factor (negative = loss)')
        ax2.set_title('Magnitude Map of Determinant')
        # ax2.grid(True, linestyle='--', alpha=0.7)

        # Add reference line at zero loss for both plots
        # ax1.axhline(y=0, color='red', linestyle='-', alpha=0.7, linewidth=1)
        # ax2.axhline(y=0, color='red', linestyle='-', alpha=0.7, linewidth=1)

        # Add a main title
        plt.suptitle('Phase and Magnitude of det(TM)', fontsize=16)
        plt.tight_layout()

        plt.show()

        return fig, (ax1, ax2)
    
    def plot_transmission_eigenvalue_spectrum(self, scatter_pos=None, freq_range=(0.45, 0.55), 
                                            freq_points=51, save_path=None):
        """
        Plot the spectrum of transmission eigenvalues (singular values) as a function of frequency.
        
        Args:
            scatter_pos: Position of scatterers (normalized), uses current positions if None
            freq_range: Tuple of (min_freq, max_freq) around self.fsrc
            freq_points: Number of frequency points to sample
            save_path: Path to save the figure (if None, display only)
            
        Returns:
            Fig, ax objects of the generated plot and the eigenvalue data
        """
        
        # Use current scatterer positions if none provided
        if scatter_pos is None:
            scatter_pos = self.scatter_pos
        
        # Create frequency array
        freqs = np.linspace(freq_range[0], freq_range[1], freq_points)
        
        # Initialize arrays to store eigenvalues at each frequency
        num_eigenvals = min(len(self.source_ports), len(self.output_ports))
        eigenval_data = np.zeros((freq_points, num_eigenvals), dtype=np.float64)
        
        # Store original frequency to restore later
        original_freq = self.fsrc
        
        # Set up progress tracking
        total_iterations = freq_points
        current_iteration = 0
        
        try:
            # Loop over frequencies
            for i, freq in enumerate(freqs):
                # Update frequency
                self.fsrc = freq
                
                # Calculate TM with current settings
                tm = self._calculate_normalized_subSM(scatter_pos, matrix_type="TM", visualize=False)
                
                # Calculate singular values (transmission eigenvalues)
                singular_values = np.linalg.svd(tm, compute_uv=False)
                eigenval_data[i, :] = singular_values
                
                # Update progress
                current_iteration += 1
                if current_iteration % 5 == 0 or current_iteration == total_iterations:
                    print(f"Progress: {current_iteration}/{total_iterations} frequencies processed")
        
        finally:
            # Restore original frequency
            self.fsrc = original_freq
        
        # Save if path provided
        if save_path:
            np.savez(save_path + 'singularvalue_spectrum.npz', 
                    freqs=freqs, 
                    eigenvalues=eigenval_data,
                    scatter_pos=scatter_pos)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot eigenvalue spectra
        for j in range(num_eigenvals):
            ax.plot(freqs, eigenval_data[:, j], '-', linewidth=2, 
                    label=f'Eigenvalue {j+1}')
        
        # Calculate and plot the Schmidt number across frequencies
        schmidt_numbers = np.zeros(freq_points)
        for i in range(freq_points):
            squared_values = eigenval_data[i, :] ** 2
            schmidt_numbers[i] = np.sum(squared_values) ** 2 / np.sum(squared_values ** 2)
        
        # Add Schmidt number on a secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(freqs, schmidt_numbers, '--', color='red', linewidth=2, label='Schmidt Number')
        ax2.set_ylabel('Schmidt Number', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([1, num_eigenvals*1.1])  # Set appropriate limits
        
        # Add labels and title
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Singular Value Magnitude')
        ax.set_title('Transmission Eigenvalue Spectrum')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Highlight regions where eigenvalues are nearly degenerate
        # Define degeneracy threshold
        degeneracy_threshold = 0.05  # 5% difference
        for i in range(freq_points):
            if np.abs(eigenval_data[i, 0] - eigenval_data[i, 1]) / eigenval_data[i, 0] < degeneracy_threshold:
                ax.axvline(x=freqs[i], color='green', alpha=0.2)
        
        # Add legends for both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # Add vertical line at reference frequency
        ax.axvline(x=self.fsrc, color='k', linestyle='--', alpha=0.7, label='Reference Freq')
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax, eigenval_data, schmidt_numbers

    def plot_schmidt_number_map(self, scatter_pos=None, scatterer_index=0, direction='x',
                                position_range=(-0.1, 0.1), position_steps=21,
                                freq_range=(0.45, 0.55), freq_steps=21, save_path=None):
        """
        Create a 2D map of Schmidt number with frequency on x-axis and 
        scatterer position shift on y-axis.
        
        Args:
            scatter_pos: Base position of scatterers (normalized), uses current positions if None
            scatterer_index: Index of the scatterer to shift (0-based)
            direction: Direction to shift the scatterer ('x' or 'y')
            position_range: Range of position shifts (normalized)
            position_steps: Number of position steps
            freq_range: Range of frequencies
            freq_steps: Number of frequency steps
            save_path: Path to save the data and figure
        
        Returns:
            fig, ax: Figure and axis objects
            schmidt_data: 2D array of Schmidt numbers
            freq_array: Array of frequencies used
            pos_array: Array of position shifts used
        """
        # Use current scatterer positions if none provided
        if scatter_pos is None:
            scatter_pos = self.scatter_pos.copy()
        else:
            scatter_pos = np.array(scatter_pos).copy()
        
        # Create arrays for frequencies and positions
        freq_array = np.linspace(freq_range[0], freq_range[1], freq_steps)
        pos_array = np.linspace(position_range[0], position_range[1], position_steps)
        
        # Determine which coordinate to modify (x or y)
        if direction.lower() == 'x':
            pos_idx = scatterer_index * 2  # x coordinate
            dir_label = "X"
        elif direction.lower() == 'y':
            pos_idx = scatterer_index * 2 + 1  # y coordinate
            dir_label = "Y"
        else:
            raise ValueError("Direction must be 'x' or 'y'")
        
        # Store original values
        original_pos = scatter_pos[pos_idx]
        original_freq = self.fsrc
        
        # Initialize array to store Schmidt numbers
        schmidt_data = np.zeros((position_steps, freq_steps))
        singular_value_data = np.zeros((position_steps, freq_steps, 2))  # For a 2x2 TM
        
        # Total iterations for progress tracking
        total_iterations = position_steps * freq_steps
        current_iteration = 0
        
        # Loop over positions and frequencies
        try:
            for p_idx, pos_shift in enumerate(pos_array):
                # Apply position shift
                scatter_pos[pos_idx] = original_pos + pos_shift
                
                # Make sure we stay in bounds
                scatter_pos[pos_idx] = np.clip(scatter_pos[pos_idx], -1, 1)
                
                for f_idx, freq in enumerate(freq_array):
                    # Update frequency
                    self.fsrc = freq
                    
                    # Calculate transmission matrix at this configuration
                    tm = self._calculate_normalized_subSM(scatter_pos, matrix_type="TM", visualize=False)
                    
                    # Calculate singular values
                    singular_values = np.linalg.svd(tm, compute_uv=False)
                    singular_value_data[p_idx, f_idx, :] = singular_values
                    
                    # Calculate Schmidt number
                    squared_values = singular_values ** 2
                    schmidt_number = np.sum(squared_values) ** 2 / np.sum(squared_values ** 2)
                    schmidt_data[p_idx, f_idx] = schmidt_number
                    
                    # Update progress
                    current_iteration += 1
                    if current_iteration % 5 == 0 or current_iteration == total_iterations:
                        print(f"Progress: {current_iteration}/{total_iterations} configurations completed")
        
        finally:
            # Restore original values
            scatter_pos[pos_idx] = original_pos
            self.fsrc = original_freq
        
        # Save data if path provided
        if save_path:
            np.savez(save_path + 'schmidt_map_data.npz',
                    freq_array=freq_array,
                    pos_array=pos_array,
                    schmidt_data=schmidt_data,
                    singular_value_data=singular_value_data,
                    scatter_pos=scatter_pos,
                    scatterer_index=scatterer_index,
                    direction=direction)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot Schmidt number map
        im = ax.pcolormesh(freq_array, pos_array, schmidt_data, 
                        cmap='viridis', shading='auto', vmin=1, vmax=2)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Schmidt Number', fontsize=12)
        
        # # Add contour lines at important Schmidt number values
        # contour_levels = [1.5, 1.75, 1.9, 1.95, 1.99]
        # contours = ax.contour(freq_array, pos_array, schmidt_data, 
        #                     levels=contour_levels, colors='white', alpha=0.5, linewidths=1)
        # ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        
        # Highlight the degeneracy line (Schmidt number ≈ 2)
        degeneracy_threshold = 1.98  # Very close to perfect degeneracy
        degeneracy_mask = schmidt_data > degeneracy_threshold
        if np.any(degeneracy_mask):
            ax.contour(freq_array, pos_array, schmidt_data, 
                    levels=[degeneracy_threshold], colors='red', linewidths=2)
        
        # Add reference point at original position and reference frequency
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax.axvline(x=original_freq, color='black', linestyle='--', alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_ylabel(f'Scatterer {scatterer_index} {dir_label}-Position Shift', fontsize=12)
        ax.set_title(f'Schmidt Number Map (Scatterer {scatterer_index} {dir_label}-Shift vs Frequency)', fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax, schmidt_data, freq_array, pos_array, singular_value_data

    def schmidt_number(self, tm):
        """
        Calculate the Schmidt number of a transmission matrix.
        
        Parameters:
        -----------
        tm : numpy.ndarray
            The transmission matrix (can be any shape)
        
        Returns:
        --------
        float
            The Schmidt number
        
        Example:
        --------
        >>> tm = np.array([[0.5, 0.5], [0.5, 0.5]])
        >>> schmidt_number(tm)
        2.0
        """
        # Calculate singular values
        singular_values = np.linalg.svd(tm, compute_uv=False)
        
        # Square the singular values
        squared_values = singular_values ** 2
        
        # Calculate Schmidt number: (sum of squared values)^2 / sum of (squared values)^2
        numerator = np.sum(squared_values) ** 2
        denominator = np.sum(squared_values ** 2)
        
        # Avoid division by zero
        if denominator == 0:
            return 0
        
        return numerator / denominator

    def _create_base_geometry(self):
        """
        Create the base geometry (billiard and waveguides).
        Child classes should implement this method.
        """
        raise NotImplementedError("Subclasses must implement _create_base_geometry")
    
    def _calculate_reward(self, tm, target_type):
        """
        Calculate reward based on the transmission matrix.
        Child classes should implement this method.
        """
        raise NotImplementedError("Subclasses must implement _calculate_reward")
    
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