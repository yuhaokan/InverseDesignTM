from stable_baselines3.common.monitor import Monitor

from ..enums import BilliardType
from ..envs import BilliardTwoEnv, BilliardThreeEnv


def make_env(billiard_type, target_type, log_dir):
    """Return a callable that creates a monitored billiard environment.

    Used by SubprocVecEnv to spawn parallel environments for SB3 training.

    Args:
        billiard_type: BilliardType enum selecting the cavity configuration.
        target_type: TargetType enum selecting the reward objective.
        log_dir: Directory for Monitor log files.
    """
    def _init():
        if billiard_type == BilliardType.TWO_PORT:
            env = BilliardTwoEnv(target_type)
        elif billiard_type == BilliardType.THREE_PORT:
            env = BilliardThreeEnv(target_type)
        else:
            raise ValueError(f"Unsupported billiard type: {billiard_type}")
        env = Monitor(env, log_dir)
        return env
    return _init
