import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import meep as mp
import typing

from gymnasium import spaces

class BilliardBaseEnv(gym.Env):
    def __init__(self):
        super().__init__()

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
        