import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SaveBestPosCallback(BaseCallback):
    """SB3 callback that saves the best scatterer positions found during training.

    On every training step, checks the reward from each parallel environment.
    Since reward = -error, a higher reward means a lower error. If a new best
    reward is found, the corresponding scatterer positions are saved to disk
    as a .npy file. Training stops early when the best reward exceeds
    ``reward_threshold`` (i.e. error drops below -reward_threshold).

    Args:
        error_threshold: Stop training when error drops below this value.
            Converted internally to reward_threshold = -error_threshold.
        save_path: Directory where the .npy checkpoint is written.
        save_name: Base filename (without extension) for the checkpoint.
    """

    def __init__(self, error_threshold, save_path, save_name):
        super().__init__()
        self.reward_threshold = -error_threshold
        self.save_path = save_path
        self.save_name = save_name
        self.best_pos = None
        self.best_reward = float('-inf')

    def _on_step(self):
        """Called after every environment step across all parallel envs.

        Returns False to stop training when reward_threshold is reached.
        """
        rewards = self.locals['rewards']
        new_obs = self.locals['new_obs']

        best_idx = np.argmax(rewards)
        if rewards[best_idx] > self.best_reward:
            self.best_reward = rewards[best_idx]
            self.best_pos = new_obs[best_idx].copy()
            self._save_best_pos()
            print(f"New best reward: {self.best_reward:.6f}")

        if self.best_reward > self.reward_threshold:
            print(f"Found solution above threshold! Reward: {self.best_reward:.6f}")
            return False

        return True

    def _save_best_pos(self):
        """Write the current best positions, reward, and step count to disk."""
        save_dict = {
            'best_pos': self.best_pos,
            'best_reward': self.best_reward,
            'n_calls': self.n_calls
        }
        np.save(os.path.join(self.save_path, self.save_name + '.npy'), save_dict)
