from typing import Tuple

import torch
import torch as th
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.distributions import Distribution
from policies.cule_bfs import CuleBFS


class ActorCriticCnnTSPolicy(ActorCriticCnnPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, tree_depth, gamma, step_env,
                 **kwargs):
        super(ActorCriticCnnTSPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
        # env_name, tree_depth, env_kwargs, gamma=0.99, step_env=None
        self.cule_bfs = CuleBFS(step_env, tree_depth, gamma)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        leaves_observations, rewards = self.cule_bfs.bfs(obs, self.cule_bfs.max_depth)
        # TODO: Check if more efficient extracting features from obs and leaves_observation simultaneously
        # Preprocess the observation if needed
        features = self.extract_features(leaves_observations)
        latent_pi, _ = self.mlp_extractor(features)
        _, latent_vf = self.mlp_extractor(self.extract_features(obs))
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        # Assaf added
        val_coef = self.cule_bfs.gamma ** self.cule_bfs.max_depth
        mean_actions = val_coef * self.action_net(latent_pi) + rewards.reshape([-1, 1])
        # # This version builds distribution to have
        mean_actions_per_subtree = mean_actions.reshape([self.action_space.n, -1])
        # TODO: Verify this numerically
        # mean_actions_logits = th.reshape(th.log(th.sum(th.exp(mean_actions_per_subtree), dim=1, keepdim=True)), (1, -1))
        mean_actions_logits = th.logsumexp(mean_actions_per_subtree, dim=1, keepdim=True).transpose(1, 0)
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        # distribution = self.action_dist.proba_distribution(action_logits=mean_actions.reshape(1, -1))
        # slice_length = self.action_space.n ** self.cule_bfs.max_depth
        # actions = distribution.get_actions(deterministic=deterministic) // slice_length
        # same_action_indexes = th.arange(slice_length*actions[0], slice_length*(actions[0]+1), device=actions.device)
        # log_prob = th.log(th.sum(th.exp(distribution.log_prob(same_action_indexes)), dim=0, keepdim=True))
        # TODO: handle fire
        return actions, values, log_prob

        # RAND_FIRE_LIST = ['Breakout']
        # fire_env = len([e for e in RAND_FIRE_LIST if e in self.env_name]) > 0
        # if fire_env and np.random.rand() < 0.01:
        #     # make sure 'FIRE' is pressed often enough to launch ball after life loss
        #     # return torch.tensor([1], device=self.device), torch.tensor(0, device=self.device)
        #     fire_pressed[0] = True
        #     return 1

    # def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    #     """
    #     Evaluate actions according to the current policy,
    #     given the observations.
    #
    #     :param obs:
    #     :param actions:
    #     :return: estimated value, log likelihood of taking those actions
    #         and entropy of the action distribution.
    #     """
    #     return super(ActorCriticCnnTSPolicy, self).evaluate_actions(obs, actions)
        # _, latent_vf = self.mlp_extractor(self.extract_features(obs))
        # values = self.value_net(latent_vf)
        # batch_size = obs.shape[0]
        # mean_actions_logits = torch.zeros((batch_size, self.action_space.n), device=values.device)
        # # Preprocess the observation if needed
        # for i in range(batch_size):
        #     leaves_observations, rewards = self.cule_bfs.bfs(obs[i:i+1, :], self.cule_bfs.max_depth)
        #     # Preprocess the observation if needed
        #     features = self.extract_features(leaves_observations)
        #     latent_pi, _ = self.mlp_extractor(features)
        #     # Assaf added
        #     val_coef = self.cule_bfs.gamma ** self.cule_bfs.max_depth
        #     mean_actions = val_coef * self.action_net(latent_pi) + rewards.reshape([-1, 1])
        #     mean_actions_per_subtree = mean_actions.reshape([-1, 1]).reshape([self.action_space.n, -1])
        #     mean_actions_logits[i, :] = th.reshape(th.log(th.sum(th.exp(mean_actions_per_subtree), dim=1, keepdim=True)), (1, -1))
        # distribution = self.action_dist.proba_distribution(action_logits=mean_actions_logits)
        # log_prob = distribution.log_prob(actions)
        # return values, log_prob, distribution.entropy()

