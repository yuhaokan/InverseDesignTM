import numpy as np
import meep as mp
import matplotlib.pyplot as plt

# Suppress logging
mp.verbosity(0)

class SingleWaveguideEnv():
    def __init__(self):
        super().__init__()

        # MEEP simulation parameters
        self.resolution = 15  # pixels/cm
        self.n_runs = 200  # number of runs during simulation

        self.fsrc = 15.0 / 30  # normalized frequency (15 GHz)

        # Waveguide parameters
        self.waveguide_width = 1.2
        self.waveguide_length = 20.0  # Total length of waveguide

        self.metal_thickness = 0.2

        # PML parameters
        self.pml_thickness = 3.0
        self.source_pml_distance = 3.0

        # Material properties
        self.epsilon_bg = 1.0  # background material (air)

        self.mode_num = 1

        self.eig_parity = mp.EVEN_Y + mp.ODD_Z

        # Define ports
        self.source_port = {
            "name": "input_port",
            "position": mp.Vector3(-self.waveguide_length / 2 + self.pml_thickness + self.source_pml_distance, 0),
            "direction": mp.X
        }

        self.output_port = {
            "name": "output_port",
            "position": mp.Vector3(self.waveguide_length / 2 - self.pml_thickness - self.source_pml_distance, 0),
            "direction": mp.X
        }

    def _create_geometry(self):
        # Create the waveguide geometry
        geometry = []

        # Top wall of waveguide
        geometry.append(mp.Block(
            material=mp.perfect_electric_conductor,
            center=mp.Vector3(0, self.waveguide_width / 2 + self.metal_thickness / 2),
            size=mp.Vector3(self.waveguide_length, self.metal_thickness)
        ))

        # Bottom wall of waveguide
        geometry.append(mp.Block(
            material=mp.perfect_electric_conductor,
            center=mp.Vector3(0, -self.waveguide_width / 2 - self.metal_thickness / 2),
            size=mp.Vector3(self.waveguide_length, self.metal_thickness)
        ))

        # Air inside the waveguide
        geometry.append(mp.Block(
            material=mp.Medium(epsilon=self.epsilon_bg),
            center=mp.Vector3(0, 0),
            size=mp.Vector3(self.waveguide_length, self.waveguide_width)
        ))

        return geometry

    def run_simulation(self, visualize=False):
        # Create geometry for the waveguide
        geometry = self._create_geometry()

        # Set up the simulation
        cell_size = mp.Vector3(self.waveguide_length, self.waveguide_width + 2 * self.metal_thickness + 4)
        pml_layers = [mp.PML(self.pml_thickness, direction=mp.X)]

        sources = [mp.EigenModeSource(
            mp.ContinuousSource(frequency=self.fsrc),
            center=self.source_port["position"],
            size=mp.Vector3(0, self.waveguide_width - 0.2),
            eig_band=self.mode_num,
            eig_parity=mp.EVEN_Y # Polarization of waveguide mode
        )]

        sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=sources,
            resolution=self.resolution,
            dimensions=2
        )

        # Add monitor at output port
        mode_monitor = sim.add_mode_monitor(
            self.fsrc, 0, 1,
            mp.ModeRegion(
                center=self.output_port["position"],
                size=mp.Vector3(0, self.waveguide_width - 0.2)
            )
        )

        # if visualize:
        #     plt.figure()
        #     sim.plot2D(plot_eps_flag=True)
        #     plt.title("Waveguide Structure")
        #     plt.show()

        # Run simulation
        sim.run(until=self.n_runs)

        # Visualize field pattern if requested
        if visualize:
            plt.figure()
            field_func = lambda x: np.sqrt(np.abs(x))
            sim.plot2D(fields=mp.Ez,
                       field_parameters={'alpha': 1, 'cmap': 'viridis', 'interpolation': 'spline36',
                                         'post_process': field_func, 'colorbar': True})
            plt.title("Electric Field")
            plt.show()

        # Calculate transmission
        mode_data = sim.get_eigenmode_coefficients(
            mode_monitor, [1],
            eig_parity=mp.EVEN_Y
        )

        transmission = abs(mode_data.alpha[0, 0, 0])

        sim.reset_meep()  # Free MEEP memory

        return transmission

    def get_transmission_spectrum(self, freq_min=0.3, freq_max=0.6, freq_points=51):
        """
        Calculate transmission spectrum over a frequency range.
        
        Args:
            freq_min: Minimum frequency (normalized)
            freq_max: Maximum frequency (normalized)
            freq_points: Number of frequency points
            visualize: Whether to visualize the spectrum
            
        Returns:
            tuple: (frequencies, transmission_values)
        """
        # Create frequency array
        frequencies = np.linspace(freq_min, freq_max, freq_points)
        transmission_values = np.zeros(freq_points, dtype=np.complex64)
        
        # Store original frequency
        original_freq = self.fsrc
        
        print("Calculating transmission spectrum...")
        
        # Loop through frequencies and calculate transmission
        for i, freq in enumerate(frequencies):
            self.fsrc = freq  # Set current frequency
            
            # Create geometry for the waveguide
            geometry = self._create_geometry()
            
            # Set up the simulation
            cell_size = mp.Vector3(self.waveguide_length, self.waveguide_width + 2 * self.metal_thickness + 4)
            pml_layers = [mp.PML(self.pml_thickness, direction=mp.X)]
            
            sources = [mp.EigenModeSource(
                mp.ContinuousSource(frequency=freq),
                center=self.source_port["position"],
                size=mp.Vector3(0, self.waveguide_width - 0.2),
                eig_band=self.mode_num,
                eig_parity=self.eig_parity  # Polarization of waveguide mode
            )]
            
            sim = mp.Simulation(
                cell_size=cell_size,
                boundary_layers=pml_layers,
                geometry=geometry,
                sources=sources,
                resolution=self.resolution,
                dimensions=2
            )
            
            # Add monitor at output port
            mode_monitor = sim.add_mode_monitor(
                freq, 0, 1,
                mp.ModeRegion(
                    center=self.output_port["position"],
                    size=mp.Vector3(0, self.waveguide_width - 0.2)
                )
            )
            
            # Run simulation
            sim.run(until=self.n_runs)
            
            # Calculate transmission
            mode_data = sim.get_eigenmode_coefficients(
                mode_monitor, [1],
                eig_parity=self.eig_parity
            )
            
            transmission_values[i] = mode_data.alpha[0, 0, 0]
            
            sim.reset_meep()  # Free MEEP memory
            
            # Print progress
            if (i + 1) % 5 == 0 or i == 0 or i == len(frequencies) - 1:
                print(f"Progress: {i + 1}/{len(frequencies)} frequencies calculated")
        
        # Restore original frequency
        self.fsrc = original_freq

        np.savez('emptyWaveguideSpectrum.npz', freqs=frequencies, transmission_values=transmission_values)
        
        return frequencies, transmission_values

if __name__ == "__main__":
    # Example usage:
    waveguide = SingleWaveguideEnv()
    transmission = waveguide.run_simulation(visualize=True)
    # print(f"Transmission: {transmission}")

    # waveguide.get_transmission_spectrum(freq_min=0.4998, freq_max=0.5002, freq_points=51)
