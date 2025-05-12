import numpy as np
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import get_linear_fn
import torch
# from torch.utils.tensorboard import SummaryWriter
# import argparse

from envs import BilliardTwoEnv, BilliardThreeEnv

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
    
class TensorboardStepCallbackV2(BaseCallback):
    def __init__(self, log_freq=32, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.last_log_step = 0
        
        # Track best reward since last logging
        self.max_reward_since_last_log = float('-inf')
        self.mean_reward_accumulator = []
        
    def _on_step(self) -> bool:
        # Get current rewards
        step_rewards = self.locals.get('rewards', [])
        
        # Always track the maximum reward and accumulate for mean, even if we don't log yet
        if len(step_rewards) > 0:
            current_max = np.max(step_rewards)
            self.max_reward_since_last_log = max(self.max_reward_since_last_log, current_max)
            self.mean_reward_accumulator.extend(step_rewards)
            
        # Only log at specified intervals
        if self.num_timesteps - self.last_log_step >= self.log_freq:
            # Log standard metrics from the model logger
            loss_dict = self.model.logger.name_to_value
            for key, value in loss_dict.items():
                self.logger.record(key, value)
            
            # Log our tracked metrics
            if self.max_reward_since_last_log > float('-inf'):
                self.logger.record("rewards/tracked_max_reward", self.max_reward_since_last_log)
            
            if len(self.mean_reward_accumulator) > 0:
                self.logger.record("rewards/tracked_mean_reward", np.mean(self.mean_reward_accumulator))
                
            # Log current episode stats if available
            if len(step_rewards) > 0:
                self.logger.record("rewards/current_step_reward_mean", np.mean(step_rewards))
                self.logger.record("rewards/current_step_reward_max", np.max(step_rewards))
            
            # Make sure to dump the logs
            self.logger.dump(self.num_timesteps)
            
            # Reset our trackers
            self.last_log_step = self.num_timesteps
            self.max_reward_since_last_log = float('-inf')
            self.mean_reward_accumulator = []
            
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
        np.save(os.path.join(self.save_path, 'best_pos_' + env_name + '_' + algo_name + '.npy'), save_dict)


def train(env_name, algo_name, error_threshold):

    # Linear learning rate decay
    lr_schedule = get_linear_fn(start=3e-4, end=1e-5, end_fraction=0.8)

    model = None

    if algo_name == "PPO":
        policy_kwargs_PPO = dict( # this policy is for PPO
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU  # ReLU often works better than tanh for physics problems
        )

        # Initialize the model
        model = PPO('MlpPolicy', env, verbose=1, device='cpu', 
                    learning_rate=lr_schedule,
                    policy_kwargs=policy_kwargs_PPO,  # Larger policy network
                    n_steps=128,    # The number of steps to run for each environment per update, 
                                    # rollout buffer size is n_steps * n_envs, 
                                    # n_steps = self.max_step // 8, or even smaller, like self.max_step // 16
                    batch_size=256, 
                    n_epochs=8,    # Number of epoch the algo will iterate through the entire collected batch of experience during each training update
                    clip_range=0.2,
                    gamma=0.999, tensorboard_log=log_dir)

    if algo_name == "SAC":
        policy_kwargs_SAC = dict( # this policy is for SAC,  SAC uses Q-functions (critics) instead of value functions, so need to use the qf key.
            net_arch=dict(
                pi=[256, 256],  # Actor/policy network architecture
                qf=[256, 256]   # Critic/Q-function network architecture
            ),
            activation_fn=torch.nn.ReLU
        )

        # With train_freq=1, gradient_steps=5, and n_envs=4:
        # Each environment step collects 4 new transitions (one from each parallel environment)
        # After collecting these 4 transitions, the agent performs 5 separate gradient updates
        # Each gradient update uses a randomly sampled batch from the entire replay buffer (not just the 4 new transitions)
        model = SAC('MlpPolicy', env, verbose=1, device='cpu', 
                learning_rate=lr_schedule,
                policy_kwargs=policy_kwargs_SAC,
                batch_size=256,
                buffer_size=100000,  # Experience replay buffer size
                train_freq=4,
                gradient_steps=8,
                learning_starts=512,  # Add this to collect enough diverse experiences before learning
                gamma=0.999,
                tau=0.005,  # For soft target updates
                ent_coef='auto',  # Automatic entropy tuning
                tensorboard_log=log_dir)

    # Create callback
    saveBestPosCallback = SaveBestPosCallback(
        error_threshold=error_threshold,
        save_freq=1000,  # Save checkpoint every 1000 steps
        verbose=1
    )
    tensorboardStepCallback = TensorboardStepCallbackV2(log_freq=32)
    try:
        # Train until we find a satisfactory solution
        model.learn(
            total_timesteps=1000000,  # Maximum steps if solution isn't found
            callback=[saveBestPosCallback, tensorboardStepCallback],
            tb_log_name=f"{env_name}_{algo_name}",
            log_interval=1
        )
    except Exception as e:
        print(f"Training stopped: {e}")
   
    if saveBestPosCallback.best_error < error_threshold:
        print("Successfully found solution:")
        print(f"Best error: {saveBestPosCallback.best_error}")
        print(f"Best position: {saveBestPosCallback.best_pos}")
        return saveBestPosCallback.best_pos
    else:
        print(f"Could not find solution below threshold. Best error: {saveBestPosCallback.best_error}")
        return saveBestPosCallback.best_pos  # Return best found even if not below threshold



# Define the environment creation function
def make_env(billiard_type):
    def _init():
        if billiard_type == 'BilliardThree':
            env = BilliardThreeEnv(target_type)
        else:
            env = BilliardTwoEnv(target_type)
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


    algo_name = "PPO"         # SAC  PPO
    billiard_type = "BilliardTwo"   # BilliardThree  BilliardTwo
    env_type = "Env12"
    target_type = "DegenerateEigVal"  # Rank1  Rank1Trace0  DegenerateEigVal

    # BilliardTwo Rank1 -> 0.1
    # BilliardThree Rank1 -> 0.02
    error_threshold = 0.02  

    env_name = billiard_type + "_" + env_type + "_" + target_type

    ## without parallel computing
    # env = BilliardTwoEnv()
    # env = Monitor(env, log_dir)
    # env = DummyVecEnv([lambda: env])


    ## Create multiple environments in parallel
    # n_envs is the number of parallel environments you want to run
    n_envs = 4  # You can adjust this number based on your CPU cores
    env = SubprocVecEnv([make_env(billiard_type) for _ in range(n_envs)])

    train(env_name, algo_name, error_threshold)
