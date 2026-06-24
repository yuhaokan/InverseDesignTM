"""Compute and plot approximate episode rewards from reward_distributions.csv.

Since episodes are always 1024 steps (max_step) and each rollout is 128 steps,
8 consecutive rollouts ≈ 1 episode. This script sums rewards across 8-rollout
windows to approximate per-episode total reward.

Note: With 4 parallel envs, episodes don't align perfectly across envs.
This gives an approximate mean episode reward per 8-rollout window.

Usage:
    python -m inverse_design.utils.plot_episode_reward
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def compute_episode_rewards(csv_path, rollouts_per_episode=8, n_envs=4):
    """Compute per-env episode rewards by summing over rollout windows.

    Args:
        csv_path: Path to reward_distributions.csv.
        rollouts_per_episode: Number of rollouts per episode (max_step / n_steps).
        n_envs: Number of parallel environments.

    Returns:
        Array of shape (n_episodes, n_envs) with per-env episode rewards.
    """
    rewards = np.loadtxt(csv_path, delimiter=',')
    n_rollouts = len(rewards)

    # Number of complete episodes we can compute
    n_episodes = n_rollouts // rollouts_per_episode

    episode_rewards = []
    for i in range(n_episodes):
        start = i * rollouts_per_episode
        end = start + rollouts_per_episode
        # Each row has n_steps * n_envs values in step-major, env-minor order
        window = rewards[start:end]  # shape: (rollouts_per_episode, n_steps * n_envs)
        # Reshape each row to (n_steps, n_envs), then sum across all steps in the window
        n_steps = window.shape[1] // n_envs
        reshaped = window.reshape(-1, n_steps, n_envs)  # (rollouts, n_steps, n_envs)
        per_env_sum = reshaped.sum(axis=(0, 1))  # shape: (n_envs,)
        episode_rewards.append(per_env_sum)

    return np.array(episode_rewards)  # shape: (n_episodes, n_envs)


def plot_episode_rewards(csv_path, save_path=None, rollouts_per_episode=8, n_envs=4):
    """Plot per-env episode reward over training.

    Args:
        csv_path: Path to reward_distributions.csv.
        save_path: If provided, save the figure to this path.
        rollouts_per_episode: Number of rollouts per episode.
        n_envs: Number of parallel environments.
    """
    episode_rewards = compute_episode_rewards(csv_path, rollouts_per_episode, n_envs)

    if len(episode_rewards) == 0:
        print("Not enough rollouts for a complete episode yet.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = range(1, len(episode_rewards) + 1)
    for env_idx in range(n_envs):
        ax.plot(episodes, episode_rewards[:, env_idx], '-o', markersize=3,
                label=f'Env {env_idx}')

    # Plot average across all envs
    avg_rewards = episode_rewards.mean(axis=1)
    ax.plot(episodes, avg_rewards, '-', linewidth=2, color='black', label='Avg')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward (sum)')
    ax.set_title('Per-Environment Episode Reward over Training')
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.3)

    # Print summary
    print(f"Total episodes: {len(episode_rewards)}")
    for env_idx in range(n_envs):
        print(f"  Env {env_idx}: latest={episode_rewards[-1, env_idx]:.4f}, "
              f"best={episode_rewards[:, env_idx].max():.4f}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'positions', 'reward_distributions_DegenerateEigVal_PPO_12.csv')

    if os.path.exists(csv_path):
        save_path = os.path.join(current_dir, '..', 'positions', 'episode_rewards.svg')
        plot_episode_rewards(csv_path, save_path=save_path)
    else:
        print(f"No reward distribution data found at {csv_path}")
