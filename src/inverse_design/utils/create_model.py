import torch
from stable_baselines3 import PPO, SAC

from ..enums import AlgoType


def create_model(algo_name, env, lr_schedule, log_dir):
    """Create an SB3 model based on the selected algorithm.

    Args:
        algo_name: AlgoType enum selecting the RL algorithm.
        env: Vectorized gymnasium environment.
        lr_schedule: Learning rate schedule function.
        log_dir: Directory for TensorBoard logs.

    Returns:
        Configured SB3 model (PPO or SAC).
    """
    if algo_name == AlgoType.PPO:
        # PPO training loop per iteration:
        #   1. Collect a rollout of n_steps * n_envs transitions using the current policy.
        #   2. Compute advantages and returns for the collected rollout.
        #   3. Shuffle the rollout into mini-batches of size batch_size.
        #   4. Update the policy n_epochs times over the full rollout,
        #      clipping the policy ratio to [1 - clip_range, 1 + clip_range].
        #   5. Discard the buffer (PPO is on-policy — no replay buffer).
        #   6. Repeat from step 1.
        #
        # Rollouts and episodes are independent:
        #   - A rollout is just "collect n_steps transitions per env, then update."
        #   - If an episode ends mid-rollout, the env resets and collection continues.
        #   - A single rollout can span multiple episodes.
        #
        # All envs share one policy and one buffer:
        #   - Transitions from all n_envs are pooled into a single buffer.
        #   - The buffer is shuffled into mini-batches for gradient updates.
        #   - Multiple envs just collect diverse data faster under the same policy.
        policy_kwargs = dict(
            net_arch=dict(
                shared=[256, 128],  # shared layer
                pi=[256, 256],      # policy (actor) head
                vf=[256, 256]       # value (critic) head
            ),
            activation_fn=torch.nn.ReLU
        )

        return PPO('MlpPolicy', env, verbose=0, device='cpu',
                    learning_rate=lr_schedule,
                    policy_kwargs=policy_kwargs,
                    n_steps=128,    # steps per env per rollout; rollout buffer = n_steps * n_envs
                    batch_size=256, # mini-batch size for each gradient update
                    n_epochs=8,     # number of passes over the rollout per iteration
                    clip_range=0.2,
                    gamma=0.999, tensorboard_log=log_dir)

    if algo_name == AlgoType.SAC:
        policy_kwargs = dict(
            net_arch=dict(
                pi=[256, 256],  # Actor/policy network
                qf=[256, 256]   # Critic/Q-function network
            ),
            activation_fn=torch.nn.ReLU
        )

        return SAC('MlpPolicy', env, verbose=1, device='cpu',
                    learning_rate=lr_schedule,
                    policy_kwargs=policy_kwargs,
                    batch_size=256,
                    buffer_size=100000,
                    train_freq=4,
                    gradient_steps=8,
                    learning_starts=512,
                    gamma=0.999,
                    tau=0.005,
                    ent_coef='auto',
                    tensorboard_log=log_dir)

    raise ValueError(f"Unsupported algorithm: {algo_name}")
