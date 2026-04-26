import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardStepCallback(BaseCallback):
    """SB3 callback that logs reward statistics to TensorBoard at fixed step intervals.

    Tracks max and mean rewards between log points and writes them alongside
    the model's own training metrics (loss, entropy, etc.).
    """
    def __init__(self, log_freq=8):
        super().__init__()
        self.log_freq = log_freq
        self.last_log_step = 0
        
        # Track best reward since last logging
        self.max_reward_since_last_log = float('-inf')
        
    def _on_step(self) -> bool:
        step_rewards = self.locals['rewards']
        
        # Track the maximum reward
        current_max = np.max(step_rewards)
        self.max_reward_since_last_log = max(self.max_reward_since_last_log, current_max)
            
        # Only log at specified intervals
        if self.n_calls - self.last_log_step >= self.log_freq:
            # Log tracked metrics
            self.logger.record("rewards/tracked_max_reward", self.max_reward_since_last_log)
            self.logger.record("rewards/current_step_reward_mean", np.mean(step_rewards))
            self.logger.record("rewards/current_step_reward_max", np.max(step_rewards))
            
            self.logger.dump(self.n_calls)
            
            # Reset trackers
            self.last_log_step = self.n_calls
            self.max_reward_since_last_log = float('-inf')
            
        return True
