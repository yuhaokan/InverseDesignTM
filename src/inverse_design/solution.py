import argparse
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import get_linear_fn
import os

from .enums import TargetType, AlgoType, BilliardType
from .callbacks import TensorboardStepCallback, SaveBestPosCallback, RewardDistributionCallback
from .utils import create_model, make_env, NetworkVisualizer


def _get_dirs():
    """Return project output directories, creating them if needed."""
    base = os.path.dirname(os.path.abspath(__file__))
    dirs = {
        "log": os.path.join(base, "logs"),
        "position": os.path.join(base, "positions"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


DIRS = _get_dirs()
log_dir = DIRS["log"]
position_dir = DIRS["position"]


def train(env, env_name, algo_name, error_threshold):
    """Train an RL agent to optimize scatterer positions.

    Configures PPO or SAC with a linear learning rate schedule, runs training
    with TensorBoard logging and best-position checkpointing, and stops early
    if the TM error drops below the threshold.

    Args:
        env: Vectorized gymnasium environment.
        env_name: Identifier string used for log and checkpoint filenames.
        algo_name: AlgoType enum selecting the RL algorithm.
        error_threshold: Training stops when TM error falls below this value.
    """

    # Linearly decay learning rate from 3e-4 to 1e-5 over the first 80% of total_timesteps,
    # then hold at 1e-5 for the remaining 20%
    lr_schedule = get_linear_fn(start=3e-4, end=1e-5, end_fraction=0.8)
    model = create_model(algo_name, env, lr_schedule, log_dir)

    # Create callback
    saveBestPosCallback = SaveBestPosCallback(
        error_threshold=error_threshold,
        save_path=position_dir,
        save_name=f"best_pos_{env_name}_{algo_name.value}",
    )
    tensorboardStepCallback = TensorboardStepCallback(log_freq=8)
    rewardDistributionCallback = RewardDistributionCallback(save_path=position_dir, n_steps=128)
    try:
        # Train until we find a satisfactory solution
        model.learn(
            total_timesteps=1000000,  # Maximum steps if solution isn't found
            callback=[saveBestPosCallback, tensorboardStepCallback, rewardDistributionCallback],
            tb_log_name=f"{env_name}_{algo_name.value}",
            log_interval=1
        )
    except Exception as e:
        print(f"Training stopped: {e}")
   
    if saveBestPosCallback.best_reward > -error_threshold:
        print(f"Found solution! Best reward: {saveBestPosCallback.best_reward:.6f}")
    else:
        print(f"Could not find solution below threshold. Best reward: {saveBestPosCallback.best_reward:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RL agent for inverse design of transmission matrices.')
    parser.add_argument('--algo', type=str, default='PPO', choices=[a.value for a in AlgoType],
                        help='RL algorithm (default: PPO)')
    parser.add_argument('--billiard', type=str, default='BilliardTwo', choices=[b.value for b in BilliardType],
                        help='Billiard cavity type (default: BilliardTwo)')
    parser.add_argument('--target', type=str, default='Rank1', choices=[t.value for t in TargetType],
                        help='Target type for reward objective (default: Rank1)')
    parser.add_argument('--error-threshold', type=float, default=0.02,
                        help='Early stopping error threshold (default: 0.02)')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments (default: 4)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the network architecture and exit (no training)')
    args = parser.parse_args()

    algo_name = AlgoType(args.algo)
    billiard_type = BilliardType(args.billiard)
    target_type = TargetType(args.target)
    error_threshold = args.error_threshold

    env_name = f"{billiard_type.value}_Env_{target_type.value}"

    env = SubprocVecEnv([make_env(billiard_type, target_type, log_dir) for _ in range(args.n_envs)])

    if args.visualize:
        lr_schedule = get_linear_fn(start=3e-4, end=1e-5, end_fraction=0.8)
        model = create_model(algo_name, env, lr_schedule, log_dir)
        NetworkVisualizer.visualize_ppo_network(model, os.path.join(log_dir, "network_graph"))
        print(f"Network graph saved to {log_dir}/network_graph")
    else:
        train(env, env_name, algo_name, error_threshold)
