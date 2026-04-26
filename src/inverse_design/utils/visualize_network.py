import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO


class NetworkVisualizer:
    """Utilities for visualizing SB3 model architectures in TensorBoard."""

    @staticmethod
    def visualize_ppo_network(model: PPO, log_dir: str):
        """Log a clean PPO network graph to TensorBoard with explicit actor/critic split.

        Wraps the policy into a custom module that clearly separates the
        feature extractor, actor (policy) pathway, and critic (value) pathway.
        Produces a readable graph in TensorBoard compared to the raw policy dump.

        Args:
            model: A trained SB3 PPO model.
            log_dir: Directory for TensorBoard log files.
        """

        class _PPONet(nn.Module):
            """Wrapper that exposes both actor and critic pathways."""
            def __init__(self, policy):
                super().__init__()
                self.features_extractor = policy.features_extractor
                self.policy_net = policy.mlp_extractor.policy_net
                self.action_net = policy.action_net
                self.value_net = policy.mlp_extractor.value_net
                self.value_head = policy.value_net

            def forward(self, obs):
                features = self.features_extractor(obs)
                pi_latent = self.policy_net(features)
                actions = self.action_net(pi_latent)
                vf_latent = self.value_net(features)
                values = self.value_head(vf_latent)
                return actions, values

        writer = SummaryWriter(log_dir)
        dummy_input = torch.zeros(1, model.observation_space.shape[0], device=model.device)
        complete_model = _PPONet(model.policy)
        writer.add_graph(complete_model, dummy_input)
        writer.close()

    @staticmethod
    def visualize_policy(model: BaseAlgorithm, log_dir: str):
        """Log the raw SB3 policy network graph to TensorBoard as-is.

        Dumps the full model.policy module including SB3's internal wrappers
        and distribution layers. Works with any SB3 algorithm but the resulting
        graph can be harder to read than the curated visualize_ppo_network output.

        Args:
            model: A trained SB3 model (any algorithm).
            log_dir: Directory for TensorBoard log files.
        """
        writer = SummaryWriter(log_dir)
        dummy_input = torch.zeros(1, model.observation_space.shape[0], device=model.device)
        # print(model.policy)
        writer.add_graph(model.policy, dummy_input)
        writer.close()
