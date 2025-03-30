import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import get_linear_fn
import torch
# from torch.utils.tensorboard import SummaryWriter
# import argparse

from envs import BilliardTwoEnv

import os
# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create directories relative to the script location
model_dir = os.path.join(current_dir, "models")
log_dir = os.path.join(current_dir, "logs")
position_dir = os.path.join(current_dir, "positions")

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
        step_rewards = self.locals['rewards']  # This is now an array with values from all environments

        # Log the mean & max step reward using logger.record
        self.logger.record("rewards/step_reward_mean", np.mean(step_rewards))
        self.logger.record("rewards/step_reward_max", np.max(step_rewards))

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
        # Check all environments in the batch
        for _, info in enumerate(self.locals['infos']):
            error = info.get('error', float('inf'))
            current_pos = info.get('scatter_pos')
            
            # Update best position if we found a better error
            if error < self.best_error and current_pos is not None:
                self.best_error = error
                self.best_pos = current_pos.copy()
                self._save_best_pos()
                print(f"New best error: {self.best_error:.6f}")

        # Stop if we find a satisfactory solution
        if self.best_error < self.error_threshold:
            print(f"Found solution below threshold! Error: {self.best_error:.6f}")
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


def train(env_name, algo_name):

    error_threshold = 0.1

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU  # ReLU often works better than tanh for physics problems
    )

    # Linear learning rate decay
    lr_schedule = get_linear_fn(start=3e-4, end=1e-5, end_fraction=0.8)
    # Initialize the model
    # batch_size=64
    model = PPO('MlpPolicy', env, verbose=1, device='cpu', 
                learning_rate=lr_schedule,
                policy_kwargs=policy_kwargs,  # Larger policy network
                n_steps=512, batch_size=128, 
                n_epochs=5, clip_range=0.2,
                gamma=0.999, tensorboard_log=log_dir)

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



# Define the environment creation function
def make_env():
    def _init():
        env = BilliardTwoEnv()
        env = Monitor(env, log_dir)
        return env
    return _init

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


    env_name = "BilliardTwoEnv2FixedTarget_pp"
    algo_name = "PPO"

    sb3_class = getattr(stable_baselines3, algo_name)


    ## without parallel computing
    # env = BilliardTwoEnv()
    # env = Monitor(env, log_dir)
    # env = DummyVecEnv([lambda: env])


    ## Create multiple environments in parallel
    # n_envs is the number of parallel environments you want to run
    n_envs = 4  # You can adjust this number based on your CPU cores
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    train(env_name, algo_name)
