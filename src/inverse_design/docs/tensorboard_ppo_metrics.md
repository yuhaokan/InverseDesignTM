# TensorBoard PPO Metrics

## Logging frequencies

- **rewards/**: Logged every `log_freq` callback calls (default 8). With `n_envs=4`, that's every 32 timesteps.
Within a rollout, the policy is frozen but the env keeps stepping — scatterer positions change each step, so rewards vary. Logging more frequently captures that variation.
- **rollout/**, **time/**: Logged every `log_interval` iterations (default 1). One iteration = one rollout = `n_steps × n_envs` = 128 × 4 = 512 timesteps.
- **train/**: Same as rollout — logged once per policy update, which happens after each rollout.

## rewards/ (custom metrics from TensorboardStepCallback)

- **tracked_max_reward**: Best reward seen across all envs since the last log point. Resets each log interval, so it shows the peak per window — useful for tracking whether the agent is finding better configurations over time.
- **current_step_reward_mean**: Mean reward across all parallel envs at the current step. Shows the average quality of the agent's current behavior.
- **current_step_reward_max**: Max reward across all parallel envs at the current step. Shows the best single-env performance at each logged step.

## rollout/ (from SB3 Monitor wrapper)

- **ep_rew_mean**: Mean total episode reward (smoothed). This is the sum of all step rewards within an episode, averaged over recent episodes. The primary indicator of training progress.
- **ep_len_mean**: Mean episode length. Since episodes only end by truncation (max_step=1024), this should stay near 1024. A drop would indicate early termination.

## time/

- **fps**: Environment steps per second. Indicates simulation throughput. Low values are expected since each MEEP simulation is computationally expensive.

## train/ (PPO internal training metrics)

- **loss**: Total loss = policy_gradient_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss. The combined objective PPO optimizes.
- **policy_gradient_loss**: The clipped surrogate objective. Measures how much the policy is being updated. Should fluctuate but not diverge.
- **value_loss**: MSE between the critic's value predictions and actual returns. Lower means the critic is better at estimating future rewards.
- **entropy_loss**: Negative entropy of the action distribution. PPO maximizes entropy (adds it as a bonus) to encourage exploration. A value approaching zero means the policy is becoming more deterministic.
- **approx_kl**: Approximate KL divergence between the old and new policy. Measures how much the policy changed in one update. Should stay small (< 0.01-0.05). Large values suggest the policy is changing too aggressively.
- **clip_fraction**: Fraction of transitions where the policy ratio was clipped by clip_range. High values (> 0.2-0.3) mean the policy is trying to change more than clip_range allows — may need a larger clip_range or smaller learning rate.
- **clip_range**: The clipping threshold (fixed at 0.2 in our config). Limits how much the policy can change per update to ensure stability.
- **explained_variance**: How well the value function explains the variance in returns. Range: (-inf, 1]. A value of 1 means perfect predictions, 0 means no better than predicting the mean, negative means worse than the mean. Should trend upward during training.
- **learning_rate**: Current learning rate from the schedule. Decays linearly from 3e-4 to 1e-5 over the first 80% of total_timesteps.
- **std**: Standard deviation of the action distribution. Starts high (exploration) and should decrease as the agent becomes more confident. If it drops too fast, the agent may get stuck in a local optimum.
