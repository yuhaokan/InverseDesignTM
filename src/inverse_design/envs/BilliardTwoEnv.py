import numpy as np
import meep as mp
import matplotlib.pyplot as plt
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
    def __init__(self, target_type="Rank1"):
        super().__init__(target_type)

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
       
    def _calculate_reward(self, tm, target_type = "Rank1") -> tuple[np.float32, np.float32]:

        match target_type:
            case "Rank1":
                # error = np.abs(tm[0][0] * tm[1][1] - tm[0][1] * tm[1][0])
                singular_values = np.linalg.svd(np.array(tm), compute_uv=False)
                ratio = singular_values[0] / np.sum(singular_values)
                error = 1 - ratio

            case "Rank1Trace0":
                # norm = np.linalg.norm(tm, 'fro')
                # if norm < 1e-10:  # Avoid division by zero
                #     norm = 1.0
                # error = np.sum(np.abs(np.linalg.eigvals(tm))) / norm

                error = (np.abs(tm[0][0] * tm[1][1] - tm[0][1] * tm[1][0]) + np.abs(tm[0][0] + tm[1][1]) ** 2) / (np.linalg.norm(tm, 'fro')**2 + 1e-8)

            case "DegenerateEigVal":
                # eigen_values = np.linalg.eigvals(tm)
                # error = np.abs(eigen_values[0] / np.sum(eigen_values) - 0.5) + np.abs(eigen_values[1] / np.sum(eigen_values) - 0.5)

                discriminant = (tm[0][0] - tm[1][1]) ** 2 + 4 * tm[0][1] * tm[1][0]
                error = np.abs(discriminant) / (np.linalg.norm(tm, 'fro') ** 2 + 1e-8) # Calculate Frobenius norm for scaling, return normalized discriminant

            case "FixedTarget":
                targetTM = np.array([[-2.28661274+0.54642883j, -7.33391126-0.31989986j], [4.91357518-2.36528964j,  3.44673878+3.01154595j]])
                error = np.sum(np.abs(tm - targetTM))

            case "DegenerateSingularVal":
                singular_values = np.linalg.svd(tm, full_matrices=False, compute_uv=False)
                error = np.abs(singular_values[0] - singular_values[1]) / (np.linalg.norm(tm, 'fro') + 1e-8)

            case _:
                error = 0

        # Reward is negative of error (higher reward for lower error)
        reward = -error
        
        return reward, error
        
    def calculate_eigenvector_coalescence(self, scatter_pos=None, freq_range=(0.45, 0.55), freq_points=21, 
                                        loss_range=(-0.05, 0.05), loss_points=21, save_path=None):
        """
        Calculate and plot eigenvector coalescence |C| as a function of frequency and loss factor.
        
        Args:
            scatter_pos: Position of scatterers (normalized), uses current positions if None
            freq_range: Tuple of (min_freq, max_freq) around self.fsrc
            freq_points: Number of frequency points to sample
            loss_range: Tuple of (min_loss, max_loss) for uniform loss
            loss_points: Number of loss points to sample
            save_path: Path to save the figure (if None, display only)
            
        Returns:
            Fig, ax objects of the generated plot and the coalescence data matrix
        """
        
        # Use current scatterer positions if none provided
        if scatter_pos is None:
            scatter_pos = self.scatter_pos
        
        # Create frequency and loss arrays
        freqs = np.linspace(freq_range[0], freq_range[1], freq_points)
        losses = np.linspace(loss_range[0], loss_range[1], loss_points)
        
        # Initialize results array for eigenvector coalescence
        coalescence_data = np.zeros((loss_points, freq_points), dtype=np.float64)
        
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
                    
                    # Calculate eigenvectors through eigendecomposition
                    eigenvalues, eigenvectors = np.linalg.eig(tm)
                    
                    # Normalize eigenvectors
                    v1 = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])
                    v2 = eigenvectors[:, 1] / np.linalg.norm(eigenvectors[:, 1])
                    
                    # Calculate eigenvector coalescence |C| as the absolute value of the inner product
                    # Following the formula from equation (5) in the paper
                    coalescence = np.abs(np.dot(v1.conj(), v2))
                    coalescence_data[i, j] = coalescence
                    
                    # Update progress
                    current_iteration += 1
                    if current_iteration % 5 == 0 or current_iteration == total_iterations:
                        print(f"Progress: {current_iteration}/{total_iterations} iterations completed")
        
        finally:
            # Restore original values
            self.fsrc = original_freq
            self.uniform_loss_factor = original_loss
        
        if save_path:
            np.savez(save_path + 'coalescence_data.npz', freqs=freqs, losses=losses, coalescence_data=coalescence_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot coalescence map
        pcm = ax.pcolormesh(
            freqs,
            -losses,
            coalescence_data,
            cmap='hot',  # Similar to heat map in paper
            vmin=0,
            vmax=1,
            shading='auto'
        )
        
        # Add colorbar
        cbar = fig.colorbar(pcm, ax=ax, label='Eigenvector Coalescence |C|')
        
        # Add labels and title
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Loss/Gain Factor')
        ax.set_title('Eigenvector Coalescence |C|')
        
        # Identify exceptional points (EPDs) where |C| approaches 1
        EPD_threshold = 0.98
        EPD_indices = np.where(coalescence_data > EPD_threshold)
        
        # Mark EPD locations with white dots
        if len(EPD_indices[0]) > 0:
            EPD_losses = losses[EPD_indices[0]]
            EPD_freqs = freqs[EPD_indices[1]]
            ax.scatter(EPD_freqs, EPD_losses, color='white', s=15, marker='o', 
                    label=f'EPDs (|C| > {EPD_threshold})', alpha=0.8)
            ax.legend()
        
        # Mark orthogonality curves where |C| approaches 0
        ortho_threshold = 0.01
        ortho_indices = np.where(coalescence_data < ortho_threshold)
        
        if len(ortho_indices[0]) > 0:
            ortho_losses = losses[ortho_indices[0]]
            ortho_freqs = freqs[ortho_indices[1]]
            ax.scatter(ortho_freqs, ortho_losses, color='white', s=5, marker='.', 
                    alpha=0.8)
        
        plt.tight_layout()
        
        plt.show()
        
        return fig, ax, coalescence_data


    def calculate_eigenvector_coalescence_position_sweep(self, scatter_pos=None, scatter_idx=0,
                                                        position_range=(-0.1, 0.1), position_points=21,
                                                        freq_range=(0.45, 0.55), freq_points=21,
                                                        direction='x', save_path=None):
        """
        Calculate and plot eigenvector coalescence |C| as a function of frequency and
        perturbation of a single scatterer's position.

        Args:
            scatter_pos: Base position of scatterers (normalized), uses current positions if None
            scatter_idx: Index of the scatterer to perturb (0 to n_scatterers-1)
            position_range: Tuple of (min_delta, max_delta) for position perturbation
            position_points: Number of position perturbation points to sample
            freq_range: Tuple of (min_freq, max_freq) around self.fsrc
            freq_points: Number of frequency points to sample
            direction: Direction to perturb the scatterer ('x' or 'y')
            save_path: Path to save the figure (if None, display only)

        Returns:
            Fig, ax objects of the generated plot and the coalescence data matrix
        """

        # Use current scatterer positions if none provided
        if scatter_pos is None:
            scatter_pos = self.scatter_pos.copy()
        else:
            scatter_pos = np.array(scatter_pos).copy()

        # Create frequency and position delta arrays
        freqs = np.linspace(freq_range[0], freq_range[1], freq_points)
        position_deltas = np.linspace(position_range[0], position_range[1], position_points)

        # Determine the actual index in the scatter_pos array based on direction
        if direction.lower() == 'x':
            pos_idx = scatter_idx * 2  # x-coordinate
        elif direction.lower() == 'y':
            pos_idx = scatter_idx * 2 + 1  # y-coordinate
        else:
            raise ValueError("Direction must be 'x' or 'y'")

        # Initialize results array for eigenvector coalescence
        coalescence_data = np.zeros((position_points, freq_points), dtype=np.float64)

        # Store original values to restore later
        original_freq = self.fsrc
        original_pos = scatter_pos[pos_idx]

        # Make sure we're not using any loss/gain
        original_loss = getattr(self, 'uniform_loss_factor', 0)
        self.uniform_loss_factor = 0

        # Set up progress tracking
        total_iterations = position_points * freq_points
        current_iteration = 0

        try:
            # Loop over position perturbations and frequencies
            for i, delta in enumerate(position_deltas):
                # Update position
                scatter_pos[pos_idx] = original_pos + delta

                # Make sure the perturbed position stays within bounds [-1, 1]
                scatter_pos[pos_idx] = np.clip(scatter_pos[pos_idx], -1, 1)

                for j, freq in enumerate(freqs):
                    # Update frequency
                    self.fsrc = freq

                    # Calculate TM with current settings
                    tm = self._calculate_subSM(scatter_pos, matrix_type="TM", visualize=False)

                    # Calculate eigenvectors through eigendecomposition
                    eigenvalues, eigenvectors = np.linalg.eig(tm)

                    # Normalize eigenvectors
                    v1 = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])
                    v2 = eigenvectors[:, 1] / np.linalg.norm(eigenvectors[:, 1])

                    # Calculate eigenvector coalescence |C|
                    coalescence = np.abs(np.dot(v1.conj(), v2))
                    coalescence_data[i, j] = coalescence

                    # Update progress
                    current_iteration += 1
                    if current_iteration % 5 == 0 or current_iteration == total_iterations:
                        print(f"Progress: {current_iteration}/{total_iterations} iterations completed")

        finally:
            # Restore original values
            self.fsrc = original_freq
            scatter_pos[pos_idx] = original_pos
            self.uniform_loss_factor = original_loss

        # Save if path provided
        if save_path:
            np.savez(save_path + 'coalescence_data.npz', freqs=freqs, position_deltas=position_deltas, coalescence_data=coalescence_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot coalescence map
        pcm = ax.pcolormesh(
            freqs,
            position_deltas,
            coalescence_data,
            cmap='hot',  # Similar to heat map in paper
            vmin=0,
            vmax=1,
            shading='auto'
        )

        # Add colorbar
        cbar = fig.colorbar(pcm, ax=ax, label='Eigenvector Coalescence |C|')

        # Add labels and title
        ax.set_xlabel('Frequency')
        ax.set_ylabel(f'Scatterer {scatter_idx} {direction.upper()}-Position Perturbation')
        ax.set_title(f'Eigenvector Coalescence vs Frequency and {direction.upper()}-Position')

        plt.tight_layout()

        plt.show()
        
        return fig, ax, coalescence_data

    def calculate_and_save_TM_sweep(self, scatter_pos=None, scatter_idx=0,
                                position_range=(-0.1, 0.1), position_points=21,
                                freq_range=(0.45, 0.55), freq_points=21,
                                direction='x', save_path=None):
        """
        Calculate and save transmission matrices (TM) as a function of frequency and
        perturbation of a single scatterer's position.

        Args:
            scatter_pos: Base position of scatterers (normalized), uses current positions if None
            scatter_idx: Index of the scatterer to perturb (0 to n_scatterers-1)
            position_range: Tuple of (min_delta, max_delta) for position perturbation
            position_points: Number of position perturbation points to sample
            freq_range: Tuple of (min_freq, max_freq) around self.fsrc
            freq_points: Number of frequency points to sample
            direction: Direction to perturb the scatterer ('x' or 'y')
            save_path: Path to save the data (required)

        Returns:
            Dictionary containing the frequency array, position delta array, and TM data
        """
        if save_path is None:
            raise ValueError("save_path must be provided to save the TM data")

        # Use current scatterer positions if none provided
        if scatter_pos is None:
            scatter_pos = self.scatter_pos.copy()
        else:
            scatter_pos = np.array(scatter_pos).copy()

        # Create frequency and position delta arrays
        freqs = np.linspace(freq_range[0], freq_range[1], freq_points)
        position_deltas = np.linspace(position_range[0], position_range[1], position_points)

        # Determine the actual index in the scatter_pos array based on direction
        if direction.lower() == 'x':
            pos_idx = scatter_idx * 2  # x-coordinate
        elif direction.lower() == 'y':
            pos_idx = scatter_idx * 2 + 1  # y-coordinate
        else:
            raise ValueError("Direction must be 'x' or 'y'")

        # Initialize results array for storing TM data
        # Each TM is a 2x2 complex matrix, so we need a 4D array
        tm_data = np.zeros((position_points, freq_points, 2, 2), dtype=np.complex128)

        # Store original values to restore later
        original_freq = self.fsrc
        original_pos = scatter_pos[pos_idx]

        # Make sure we're not using any loss/gain
        original_loss = getattr(self, 'uniform_loss_factor', 0)
        self.uniform_loss_factor = 0

        # Set up progress tracking
        total_iterations = position_points * freq_points
        current_iteration = 0

        try:
            # Loop over position perturbations and frequencies
            for i, delta in enumerate(position_deltas):
                # Update position
                scatter_pos[pos_idx] = original_pos + delta

                # Make sure the perturbed position stays within bounds [-1, 1]
                scatter_pos[pos_idx] = np.clip(scatter_pos[pos_idx], -1, 1)

                for j, freq in enumerate(freqs):
                    # Update frequency
                    self.fsrc = freq

                    # Calculate TM with current settings
                    tm = self._calculate_normalized_subSM(scatter_pos, matrix_type="TM", visualize=False)
                    
                    # Store the TM
                    tm_data[i, j] = tm

                    # Update progress
                    current_iteration += 1
                    if current_iteration % 5 == 0 or current_iteration == total_iterations:
                        print(f"Progress: {current_iteration}/{total_iterations} iterations completed")

        finally:
            # Restore original values
            self.fsrc = original_freq
            scatter_pos[pos_idx] = original_pos
            self.uniform_loss_factor = original_loss

        # Save data
        save_data = {
            'freqs': freqs,
            'position_deltas': position_deltas,
            'tm_data': tm_data,
            'scatter_idx': scatter_idx,
            'direction': direction,
            'original_position': original_pos
        }
        

        np.savez(save_path + 'tm_sweep.npz', **save_data)
        print(f"TM data saved to {save_path}")

        return save_data

    def get_eigenvectors_degenerate_case(self, scatter_pos=None):
        """
        Extracts the eigenvector and generalized eigenvector for a 2x2 matrix 
        with degenerate eigenvalues.
        
        Args:
            scatter_pos: Position of scatterers (normalized), uses current positions if None
            
        Returns:
            tuple: (eigenvalue, eigenvector, generalized_eigenvector)
        """
        # Use current scatterer positions if none provided
        if scatter_pos is None:
            scatter_pos = self.scatter_pos
            
        # Calculate the transmission matrix
        tm = self._calculate_normalized_subSM(scatter_pos, matrix_type="TM", visualize=False)
        
        # Get eigenvalues and eigenvectors from numpy
        eigenvalues, eigenvectors = np.linalg.eig(tm)
        
        # Use the average as the single eigenvalue
        eigenvalue = np.mean(eigenvalues)
        eigenvector = eigenvectors[:, 0]

        # Normalized eigenvector
        eigenvector = eigenvector/np.linalg.norm(eigenvector)

        # generalized_eigenvector = np.array([-eigenvector[1].conj(), eigenvector[0].conj()])

        # For degenerate case, we need to solve for the generalized eigenvector
        # The standard eigenvector is in the null space of (A - λI)
        A_minus_lambdaI = tm - eigenvalue * np.eye(2)

        generalized_eigenvector = np.linalg.lstsq(A_minus_lambdaI, eigenvector, rcond=None)[0]

        # Make them orthogonal
        # generalized_eigenvector = generalized_eigenvector - eigenvector.conj().T @ generalized_eigenvector * eigenvector

       
        # Perform a final check
        check1 = np.linalg.norm(A_minus_lambdaI @ eigenvector)
        check2 = np.linalg.norm(A_minus_lambdaI @ generalized_eigenvector - eigenvector)
        check3 = np.linalg.norm(eigenvector.T.conj() @ generalized_eigenvector)
        
        print(f"Verification check for eigenvector: ||(A-λI)v|| = {check1}")
        print(f"Verification check for generalized eigenvector: ||(A-λI)g - v|| = {check2}")
        print(f"Verification check for orthogonality <v|g> = {check3}")
        
        P = np.column_stack([eigenvector, generalized_eigenvector])
        print(P)
        J = np.linalg.inv(P) @ tm @ P

        print(J)

        return eigenvalue, eigenvector, generalized_eigenvector

    def get_P(self, scatter_pos=None):
        """
        Extracts the eigenvector and generalized eigenvector for a 2x2 matrix 
        with degenerate eigenvalues.
        
        Args:
            scatter_pos: Position of scatterers (normalized), uses current positions if None
            
        Returns:
            tuple: (eigenvalue, eigenvector, generalized_eigenvector)
        """
        # Use current scatterer positions if none provided
        if scatter_pos is None:
            scatter_pos = self.scatter_pos
            
        # Calculate the transmission matrix
        tm = self._calculate_normalized_subSM(scatter_pos, matrix_type="TM", visualize=False)
        
        # Get eigenvalues and eigenvectors from numpy
        eigenvalues, eigenvectors = np.linalg.eig(tm)
        
        # Use the average as the single eigenvalue
        eigenvalue = np.mean(eigenvalues)
        eigenvector = eigenvectors[:, 0]

        # Normalized eigenvector
        eigenvector = eigenvector/np.linalg.norm(eigenvector)

        # For degenerate case, we need to solve for the generalized eigenvector
        # The standard eigenvector is in the null space of (A - λI)
        A_minus_lambdaI = tm - eigenvalue * np.eye(2)

        generalized_eigenvector = np.linalg.lstsq(A_minus_lambdaI, eigenvector, rcond=None)[0]
        generalized_eigenvector = generalized_eigenvector - eigenvector.conj().T @ generalized_eigenvector * eigenvector
        
        P = np.column_stack([eigenvector, generalized_eigenvector])

        C01 = np.abs(generalized_eigenvector.conj().T @ tm @ eigenvector)**2
        C10 = np.abs(eigenvector.conj().T @ tm @ generalized_eigenvector)**2
        print(C01, C10)

        return P

    def get_Jordan_near_EP(self, P, scatter_pos=None):
        """
        Extracts the eigenvector and generalized eigenvector for a 2x2 matrix 
        with degenerate eigenvalues.
        
        Args:
            scatter_pos: Position of scatterers (normalized), uses current positions if None
            
        Returns:
            tuple: (eigenvalue, eigenvector, generalized_eigenvector)
        """
        # Use current scatterer positions if none provided
        if scatter_pos is None:
            scatter_pos = self.scatter_pos
            
        # Calculate the transmission matrix
        tm = self._calculate_normalized_subSM(scatter_pos, matrix_type="TM", visualize=False)
        
        # P for best_pos_BilliardTwo_Env12_DegenerateEigVal_PPO_8
        # P = np.array([[  0.89466086+0j,             26.24562364+92.36013569j ],
        #               [ -0.41623016+0.16227879j,    -28.96327786-38.2088546j]])

        
        J = np.linalg.inv(P) @ tm @ P

        return J

    
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
