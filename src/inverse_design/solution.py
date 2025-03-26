import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
# from torch.utils.tensorboard import SummaryWriter
import os
# import argparse

from envs import BilliardTwoEnv

# Create directories to hold models and logs
model_dir = "/home/user/workplace/InverseDesignTM/src/inverse_design/models"
log_dir = "/home/user/workplace/InverseDesignTM/src/inverse_design/logs"
position_dir = "/home/user/workplace/InverseDesignTM/src/inverse_design/positions"
os.makedirs(position_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

'''
tensorboard --logdir ./logs
python load_pos.py
'''

from stable_baselines3.common.callbacks import BaseCallback
    
class TensorboardStepCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log losses and metrics at each step
        loss_dict = self.model.logger.name_to_value
       
        for key, value in loss_dict.items():
            self.logger.record(key, value)
           
        # Get the most recent reward from the environment
        # The reward from the last step is stored in self.locals
        step_reward = self.locals['rewards'][0]  # [0] because we're using VecEnv

        # Log the step reward using logger.record
        self.logger.record("rewards/step_reward", step_reward)

        # Make sure to dump the logs
        self.logger.dump(self.num_timesteps)

        return True

class SaveBestPosCallback(BaseCallback):
    def __init__(self, error_threshold=0.5, save_freq=100, save_path=position_dir, verbose=0):
        super().__init__(verbose)
        self.error_threshold = error_threshold
        self.save_freq = save_freq
        self.save_path = save_path
        self.best_pos = None
        self.best_error = float('inf')

    def _on_step(self):
        # Get current error and position from info
        error = self.locals['infos'][0].get('error', float('inf'))
        current_pos = self.locals['infos'][0].get('scatter_pos')
       
        # Update best position if we found a better error
        if error < self.best_error:
            self.best_error = error
            self.best_pos = current_pos
           
            # Save the new best position
            self._save_best_pos()
           
            # if self.verbose > 0:
            #     print(f"New best error: {self.best_error:.6f}")
            #     print(f"New best position: {self.best_pos}")

        # Save periodically regardless of improvement
        # if self.n_calls % self.save_freq == 0:
        #     self._save_checkpoint()

        # Stop if we find a satisfactory solution
        if error < self.error_threshold:
            print(f"Found solution below threshold! Error: {error:.6f}")
            return False

        return True

    def _save_best_pos(self):
        """Save the best position found so far"""
        save_dict = {
            'best_pos': self.best_pos,
            'best_error': self.best_error,
            'n_calls': self.n_calls
        }
        np.save(os.path.join(self.save_path, 'best_pos.npy'), save_dict)

    def _save_checkpoint(self):
        """Save periodic checkpoint with timestamp"""
        save_dict = {
            'best_pos': self.best_pos,
            'best_error': self.best_error,
            'n_calls': self.n_calls
        }
        checkpoint_path = os.path.join(
            self.save_path,
            f'checkpoint_{self.n_calls:08d}.npy'
        )
        np.save(checkpoint_path, save_dict)



def train(env_name, algo_name):

    error_threshold = 0.1

    # Initialize the model
    model = sb3_class('MlpPolicy', env, verbose=1, device='cpu', gamma=0, tensorboard_log=log_dir)

    # Create callback
    callback = SaveBestPosCallback(
        error_threshold=error_threshold,
        save_freq=1000,  # Save checkpoint every 1000 steps
        verbose=1
    )
    tensorboardStepCallback = TensorboardStepCallback()
    try:
        # Train until we find a satisfactory solution
        model.learn(
            total_timesteps=1000000,  # Maximum steps if solution isn't found
            callback=[callback, tensorboardStepCallback],
            tb_log_name=f"{env_name}_{algo_name}"
        )
    except Exception as e:
        print(f"Training stopped: {e}")
   
    if callback.best_error < error_threshold:
        print("Successfully found solution:")
        print(f"Best error: {callback.best_error}")
        print(f"Best position: {callback.best_pos}")
        return callback.best_pos
    else:
        print(f"Could not find solution below threshold. Best error: {callback.best_error}")
        return callback.best_pos  # Return best found even if not below threshold


if __name__ == '__main__':
    # Parse command line inputs
    # parser = argparse.ArgumentParser(description='Train or test model.')
    # parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    # parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. A2C, DDPG, DQN, PPO, SAC, TD3')    
    # parser.add_argument('--test', help='Test mode', action='store_true')
    # args = parser.parse_args()

    # Dynamic way to import algorithm. For example, passing in DQN is equivalent to hardcoding:
    # from stable_baselines3 import DQN
    # sb3_class = getattr(stable_baselines3, args.sb3_algo)


    env_name = "BilliardTwoEnvFixedTarget"
    algo_name = "PPO"

    sb3_class = getattr(stable_baselines3, algo_name)

    # env = gym.make(env_name)

    env = BilliardTwoEnv()
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    # env = gym.wrappers.RecordVideo(env, video_folder=recording_dir, episode_trigger = lambda x: x % 10000 == 0)
    train(env_name, algo_name)
