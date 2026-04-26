from enum import Enum


class AlgoType(str, Enum):
    """Reinforcement learning algorithms supported for training."""

    PPO = "PPO"
    """Proximal Policy Optimization."""

    SAC = "SAC"
    """Soft Actor-Critic."""
