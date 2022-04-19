from typing import Tuple

import torch
import torch as th
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.distributions import Distribution
from policies.cule_bfs import CuleBFS


class ActorCriticCnnTSPolicy(ActorCriticCnnPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, tree_depth, gamma, step_env, buffer_size,
                 learn_alpha, use_tree_for_v, **kwargs):
        super(ActorCriticCnnTSPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
        # env_name, tree_depth, env_kwargs, gamma=0.99, step_env=None
        self.cule_bfs = CuleBFS(step_env, tree_depth, gamma, self.compute_value)
        self.time_step = 0
        self.obs2leaves_dict = {}
        self.timestep2obs_dict = {}
        self.obs2timestep_dict = {}
        self.use_tree_for_v = use_tree_for_v
        # self.obs_dict = {}
        self.buffer_size = buffer_size
        self.learn_alpha = learn_alpha
        self.alpha = th.tensor(1.0 * (action_space.n ** tree_depth) , device=self.device)
        if self.learn_alpha:
            self.alpha = th.nn.Parameter(self.alpha)
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        hash_obs = self.hash_obs(obs)[0].item()
        # self.obs_dict[hash_obs] = 1
        if hash_obs in self.obs2leaves_dict:
            leaves_observations, rewards = self.obs2leaves_dict.get(hash_obs)
            del self.timestep2obs_dict[self.obs2timestep_dict[hash_obs]]
        else:
            leaves_observations, rewards = self.cule_bfs.bfs(obs, self.cule_bfs.max_depth)
            self.obs2leaves_dict[hash_obs] = leaves_observations, rewards
        self.obs2timestep_dict[hash_obs] = self.time_step
        self.timestep2obs_dict[self.time_step] = hash_obs
        # TODO: Check if more efficient extracting features from obs and leaves_observation simultaneously
        # Preprocess the observation if needed
        latent_pi, value_root = self.compute_value(root_obs=obs, leaves_obs=leaves_observations)
        val_coef = self.cule_bfs.gamma ** self.cule_bfs.max_depth
        mean_actions = val_coef * self.action_net(latent_pi) + rewards.reshape([-1, 1])
        # values = (val_coef * self.value_net(latent_vf) + rewards.reshape([-1, 1])).max(0, keepdims=True)[0]
        # # This version builds distribution to have
        mean_actions_per_subtree = self.alpha * mean_actions.reshape([self.action_space.n, -1])
        # TODO: Verify this numerically
        # mean_actions_logits = th.reshape(th.log(th.sum(th.exp(mean_actions_per_subtree), dim=1, keepdim=True)), (1, -1))
        mean_actions_logits = th.logsumexp(mean_actions_per_subtree, dim=1, keepdim=True).transpose(1, 0)
        # mean_actions_logits = th.mean(mean_actions_per_subtree, dim=1, keepdim=True).transpose(1, 0)
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        # DEBUG:
        # actions = th.randint(0, self.action_space.n, (1, ), device=actions.device)
        log_prob = distribution.log_prob(actions)
        # TODO: handle fire
        if self.time_step - self.buffer_size in self.timestep2obs_dict:
            del self.obs2leaves_dict[self.timestep2obs_dict[self.time_step - self.buffer_size]]
            del self.obs2timestep_dict[self.timestep2obs_dict[self.time_step - self.buffer_size]]
            del self.timestep2obs_dict[self.time_step - self.buffer_size]
        self.time_step += 1
        # print("New observations: ", len(self.obs_dict)/self.time_step, len(self.obs_dict), self.time_step)
        return actions, value_root, log_prob

        # RAND_FIRE_LIST = ['Breakout']
        # fire_env = len([e for e in RAND_FIRE_LIST if e in self.env_name]) > 0
        # if fire_env and np.random.rand() < 0.01:
        #     # make sure 'FIRE' is pressed often enough to launch ball after life loss
        #     # return torch.tensor([1], device=self.device), torch.tensor(0, device=self.device)
        #     fire_pressed[0] = True
        #     return 1

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        batch_size = obs.shape[0]
        mean_actions_logits = torch.zeros((batch_size, self.action_space.n), device=actions.device)
        # Preprocess the observation if needed
        hash_obses = self.hash_obs(obs)
        all_leaves_obs = [obs]
        all_rewards = []
        for i in range(batch_size):
            hash_obs = hash_obses[i].item()
            if hash_obs in self.obs2leaves_dict:
                leaves_observations, rewards = self.obs2leaves_dict.get(hash_obs)
            else:
                print("This shouldn't happen! observation not in our dictionary")
                leaves_observations, rewards = self.cule_bfs.bfs(obs, self.cule_bfs.max_depth)
                self.obs2leaves_dict[hash_obs] = leaves_observations, rewards
            all_leaves_obs.append(leaves_observations)
            all_rewards.append(rewards)
            # Preprocess the observation if needed
        all_rewards_th = th.cat(all_rewards).reshape([-1, 1])
        val_coef = self.cule_bfs.gamma ** self.cule_bfs.max_depth
        cat_features = self.extract_features(th.cat(all_leaves_obs))
        shared_features = self.mlp_extractor.shared_net(cat_features)
        latent_pi = self.mlp_extractor.policy_net(shared_features[batch_size:])
        latent_vf_root = self.mlp_extractor.value_net(shared_features[:batch_size])
        values = self.value_net(latent_vf_root)
        # _, latent_vf = self.mlp_extractor(self.extract_features(obs))
        # values = self.value_net(latent_vf)
        # features = self.extract_features(th.cat(all_leaves_obs))
        # latent_pi, latent_vf = self.mlp_extractor(features)
        # Assaf added
        mean_actions = val_coef * self.action_net(latent_pi) + all_rewards_th
        subtree_width = self.action_space.n**self.cule_bfs.max_depth
        # mean_vf = val_coef * self.value_net(latent_vf) + all_rewards_th
        # values = mean_vf.reshape((-1, batch_size, 1)).max(dim=0)[0]
        # mean_actions.reshape((-1, batch_size, self.action_space.n)).transpose(0, 1).reshape([batch_size, self.action_space.n, -1])
        for i in range(batch_size):
            mean_actions_per_subtree = self.alpha * mean_actions[subtree_width*i:subtree_width*(i+1)].reshape([self.action_space.n, -1])
            mean_actions_logits[i, :] = th.logsumexp(mean_actions_per_subtree, dim=1, keepdim=True).transpose(1, 0)
            # mean_actions_logits[i, :] = th.mean(mean_actions_per_subtree, dim=1, keepdim=True).transpose(1, 0)
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions_logits)
        log_prob = distribution.log_prob(actions)
        # old_values, old_log_probs, old_dist = super(ActorCriticCnnPolicy, self).evaluate_actions(obs, actions)
        # print("Bla", old_values - values, old_log_probs - log_prob, old_dist - distribution.entropy())
        return values, log_prob, distribution.entropy()

    def hash_obs(self, obs):
        return (obs[:, -2:, :, :].double() + obs[:, -2:, :, :].double().pow(2)).view(obs.shape[0], -1).sum(dim=1).int()

    def compute_value(self, root_obs, leaves_obs):
        cat_features = self.extract_features(th.cat((root_obs, leaves_obs)))
        shared_features = self.mlp_extractor.shared_net(cat_features)
        latent_pi = self.mlp_extractor.policy_net(shared_features[1:])
        latent_vf_root = self.mlp_extractor.value_net(shared_features[:1])
        value_root = self.value_net(latent_vf_root)
        return latent_pi, value_root