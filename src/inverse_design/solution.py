import gymnasium as gym
import stable_baselines3
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os
# import argparse

from envs import BilliardTwoEnv

# Create directories to hold models and logs
model_dir = "./models"
log_dir = "./logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


# from stable_baselines3.common.callbacks import BaseCallback

# class DebugCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super().__init__(verbose)
#         self.episode_count = 0

#     def _on_step(self) -> bool:
#         # Get episode info from monitor
#         if len(self.training_env.get_attr('n_scatterers')) > 0:
#             last_reward = self.training_env.get_attr('n_scatterers')[-1]
#             last_length = self.training_env.get_attr('n_scatterers')[-1]
           
#             # Log the metrics
#             self.logger.record("debug/episode_reward", last_reward)
#             self.logger.record("debug/episode_length", last_length)
#             print(f"Episode {len(self.training_env.get_attr('n_scatterers'))}: "
#                   f"Reward = {last_reward}, Length = {last_length}")  # Debug print

#         return True
    

def train(env_name, algo_name):
    model = sb3_class('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    # Stop training when mean reward reaches reward_threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-1, verbose=1)

    # Stop training when model shows no improvement after max_no_improvement_evals, 
    # but do not start counting towards max_no_improvement_evals until after min_evals.
    # Number of timesteps before possibly stopping training = min_evals * eval_freq (below)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=20, verbose=1)

    # debug_callback = DebugCallback()

    eval_callback = EvalCallback(
        env, 
        eval_freq=25, # how often to perform evaluation i.e. every 10000 timesteps.
        callback_on_new_best=callback_on_best, 
        callback_after_eval=stop_train_callback, 
        verbose=1, 
        best_model_save_path=os.path.join(model_dir, f"{env_name}_{algo_name}"),
    )
    
    """
    total_timesteps: pass in a very large number to train (almost) indefinitely.
    tb_log_name: create log files with the name [gym env name]_[sb3 algorithm] i.e. Pendulum_v1_SAC
    callback: pass in reference to a callback fuction above
    """
    model.learn(total_timesteps=int(1e7), tb_log_name=f"{env_name}_{algo_name}", callback=eval_callback)


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


    env_name = "BilliardTwoEnv"
    algo_name = "SAC"

    sb3_class = getattr(stable_baselines3, algo_name)

    # env = gym.make(env_name)

    env = BilliardTwoEnv()
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    # env = gym.wrappers.RecordVideo(env, video_folder=recording_dir, episode_trigger = lambda x: x % 10000 == 0)
    train(env_name, algo_name)
