import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df_mean_reward = pd.read_csv('./BilliardTwo_Env12_DegenerateEigVal_PPO_6_tracked_mean_reward.csv')
df_max_reward = pd.read_csv('./BilliardTwo_Env12_DegenerateEigVal_PPO_6_tracked_max_reward.csv')
df_ep_mean_reward = pd.read_csv('./BilliardTwo_Env12_DegenerateEigVal_PPO_6_ep_rew_mean.csv')
# Calculate the running maximum for max rewards
df_max_reward['RunningMax'] = df_max_reward['Value'].cummax()


# Create the figure and axis
f = lambda x: -np.log10(-x)

# Create figure with two vertically stacked subplots sharing the x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Upper panel: Max reward and running max after transformation
ax1.scatter(df_max_reward['Step'], f(df_max_reward['Value']),
           marker='o', color='#1f77b4', alpha=0.7, label='Max Reward')
ax1.scatter(df_mean_reward['Step'], f(df_mean_reward['Value']),
           marker='o', color="#1fb45d", alpha=0.7, label='Mean Reward')
ax1.plot(df_max_reward['Step'], f(df_max_reward['RunningMax']),
         color='red', linewidth=2, label='Running Max')
ax1.set_ylabel('Transformed Value (-log(-x))', fontsize=12)
ax1.set_title('Transformed Max Rewards', fontsize=14)
ax1.legend()
ax1.grid(alpha=0.3)

# Lower panel: Mean reward
ax2.plot(df_ep_mean_reward['Step'][::8], df_ep_mean_reward['Value'][::8],
         color='green', linewidth=2, label='Mean ep Reward')
ax2.set_xlabel('Step', fontsize=12)
ax2.set_ylabel('Mean Reward', fontsize=12)
ax2.set_title('Mean Rewards Over Time', fontsize=14)
ax2.legend()
ax2.grid(alpha=0.3)

# Improve layout with padding
plt.tight_layout()
# plt.subplots_adjust(hspace=0.3)  # Add some space between the subplots

plt.show()