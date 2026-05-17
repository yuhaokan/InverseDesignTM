import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardDistributionCallback(BaseCallback):
    """SB3 callback that appends per-rollout raw reward data to a CSV file.

    Collects all rewards from every step within a rollout, then appends one row
    to the CSV at the end of each rollout. Each row contains n_steps * n_envs
    reward values. Use the CSV for custom plotting (e.g. ridgeline plots).

    Args:
        save_path: Directory to save the reward CSV file.
        n_steps: Number of steps per rollout (must match PPO's n_steps).
    """

    def __init__(self, save_path, n_steps=128):
        super().__init__()
        self.csv_path = os.path.join(save_path, 'reward_distributions.csv')
        self.n_steps = n_steps
        self.rollout_rewards = []

        # Clear the file at the start of training
        open(self.csv_path, 'w').close()

    def _on_step(self) -> bool:
        rewards = self.locals['rewards']
        self.rollout_rewards.extend(rewards)

        # Append to CSV at the end of each rollout
        if self.n_calls % self.n_steps == 0:
            with open(self.csv_path, 'a') as f:
                f.write(','.join(f'{r:.6f}' for r in self.rollout_rewards) + '\n')
            self.rollout_rewards = []

        return True
