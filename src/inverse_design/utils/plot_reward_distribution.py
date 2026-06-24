"""Plot ridgeline (joy plot) of reward distributions across rollouts.

Usage:
    python -m inverse_design.utils.plot_reward_distribution
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_ridgeline(csv_path, save_path=None, step=1):
    """Plot a ridgeline plot showing reward distribution evolution over training.

    Args:
        csv_path: Path to reward_distributions.csv (one row per rollout, 512 values each).
        save_path: If provided, save the figure to this path.
        step: Plot every Nth rollout (default 1 = all rollouts).
    """
    rewards = np.loadtxt(csv_path, delimiter=',')
    n_rollouts = len(rewards)

    # Select rollouts to plot
    indices = range(0, n_rollouts, step)
    n_plots = len(list(indices))

    fig, axes = plt.subplots(n_plots, 1, figsize=(8, n_plots * 0.6), sharex=True)
    if n_plots == 1:
        axes = [axes]

    x_min = rewards.min()
    x_max = rewards.max()
    x_range = np.linspace(x_min, x_max, 200)

    for i, rollout_idx in enumerate(range(0, n_rollouts, step)):
        ax = axes[i]
        data = rewards[rollout_idx]

        # Compute KDE
        kde = gaussian_kde(data, bw_method=0.05)
        density = kde(x_range)

        ax.fill_between(x_range, density, alpha=0.6, color=plt.cm.viridis(rollout_idx / n_rollouts))
        ax.plot(x_range, density, color='black', linewidth=0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_yticks([])
        ax.set_ylabel(f'{rollout_idx + 1}', rotation=0, labelpad=20, fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axes[-1].set_xlabel('Reward')
    fig.supylabel('Rollout idx')
    # fig.suptitle('Reward Distribution per Rollout', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'positions', 'reward_distributions.csv')

    if os.path.exists(csv_path):
        save_path = os.path.join(current_dir, '..', 'positions', 'reward_ridgeline3_0.05_3.svg')
        plot_ridgeline(csv_path, save_path=save_path, step=3)
    else:
        print(f"No reward distribution data found at {csv_path}")
