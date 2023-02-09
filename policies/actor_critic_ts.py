from typing import Tuple

import torch
import torch as th
import numpy as np
import math
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.distributions import Distribution

from policies.actor_critic_depth0 import ActorCriticCnnPolicyDepth0
from policies.cule_bfs import CuleBFS
from utils import add_regularization_logits


class ActorCriticCnnTSPolicy(ActorCriticCnnPolicyDepth0):
    def __init__(self, observation_space, action_space, lr_schedule, tree_depth, gamma, step_env, buffer_size,
                 learn_alpha, learn_beta, max_width, use_leaves_v, is_cumulative_mode, regularization, **kwargs):
        super(ActorCriticCnnTSPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
        # env_name, tree_depth, env_kwargs, gamma=0.99, step_env=None
        self.cule_bfs = CuleBFS(step_env, tree_depth, gamma, self.compute_value, max_width)
        self.time_step = 0
        self.obs2leaves_dict = {}
        self.timestep2obs_dict = {}
        self.obs2timestep_dict = {}
        # self.obs_dict = {}
        self.buffer_size = buffer_size
        self.learn_alpha = learn_alpha
        self.learn_beta = learn_beta
        self.is_cumulative_mode = is_cumulative_mode
        self.regularization = regularization
        self.alpha = th.tensor(0.5 if learn_alpha else 1.0, device=self.device)
        self.beta = th.tensor(1.0, device=self.device)
        # if max_width == -1:
        #     self.alpha = th.tensor(1.0 * action_space.n ** tree_depth, device=self.device)
        # else:
        #     self.alpha = th.tensor(1.0 * min(action_space.n ** tree_depth, max_width), device=self.device)
        if self.learn_alpha:
            self.alpha = th.nn.Parameter(self.alpha)
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        if self.learn_beta:
            self.beta = th.nn.Parameter(self.beta)
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.use_leaves_v = use_leaves_v

    # @profile
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
            leaves_observations, rewards, first_action = self.obs2leaves_dict.get(hash_obs)
            del self.timestep2obs_dict[self.obs2timestep_dict[hash_obs]]
        else:
            leaves_observations, rewards, first_action = self.cule_bfs.bfs(obs, self.cule_bfs.max_depth)
            self.obs2leaves_dict[hash_obs] = leaves_observations, rewards, first_action
        self.obs2timestep_dict[hash_obs] = self.time_step
        self.timestep2obs_dict[self.time_step] = hash_obs
        # TODO: Check if more efficient extracting features from obs and leaves_observation simultaneously
        # Preprocess the observation if needed
        val_coef = self.cule_bfs.gamma ** self.cule_bfs.max_depth
        if self.use_leaves_v:
            latent_pi, value_root = self.compute_value(leaves_obs=leaves_observations, root_obs=obs)
            value_root = (val_coef * value_root + rewards.reshape([-1, 1])).max()
        else:
            latent_pi, value_root = self.compute_value2(leaves_obs=leaves_observations, root_obs=obs)
        mean_actions = val_coef * self.action_net(latent_pi) + rewards.reshape([-1, 1])

        # values = (val_coef * self.value_net(latent_vf) + rewards.reshape([-1, 1])).max(0, keepdims=True)[0]
        # # This version builds distribution to have
        if self.cule_bfs.max_width == -1:
            mean_actions_per_subtree = self.beta * mean_actions.reshape([self.action_space.n, -1])
            counts = th.ones([1, self.action_space.n]) * mean_actions_per_subtree.shape[1]
        else:
            mean_actions_per_subtree = th.zeros(self.action_space.n, mean_actions.shape[0], mean_actions.shape[1],
                                                device=mean_actions.device) # -1e6
            idxes = th.arange(mean_actions.shape[0])
            counts = th.zeros(self.action_space.n)
            v, c = th.unique(first_action, return_counts=True)
            counts[v] = c.type(th.float32) * self.action_space.n
            mean_actions_per_subtree[first_action.flatten(), idxes, :] = mean_actions
            mean_actions_per_subtree = self.beta * mean_actions_per_subtree.reshape([self.action_space.n, -1])
        counts = counts.to(mean_actions.device).reshape([1, -1])
        if self.is_cumulative_mode:
            mean_actions_logits = th.sum(mean_actions_per_subtree, dim=1, keepdim=True).transpose(1, 0) / counts
        else:
            # To obtain the mean we subtract the normalization log(#leaves)
            mean_actions_logits = th.logsumexp(mean_actions_per_subtree, dim=1, keepdim=True).transpose(1, 0) - \
                                  th.log(counts)#math.log(mean_actions_per_subtree.shape[1])
        mean_actions_logits[counts == 0] = -math.inf

            # mean_actions_logits2 = torch.zeros(1, self.action_space.n, device=mean_actions.device)
            # for action in range(self.action_space.n):
            #     mean_actions_logits2[0, action] = th.logsumexp(self.beta * mean_actions[first_action.flatten() == action, :].flatten(), dim=0, keepdim=True)
            #
            # squash_q = th.sum(th.clip(th.exp(self.beta * mean_actions), 0, 1 / (1 - self.cule_bfs.gamma)), dim=1, keepdim=True)
            # mean_actions_logits = torch.zeros(self.action_space.n, 1, device=squash_q.device)
            # mean_actions_logits.scatter_add_(0, first_action.to(squash_q.device), squash_q)
            # mean_actions_logits = torch.log(mean_actions_logits.transpose(1, 0) + 1e-6)
            # t2 = time.time()
            # for t in range(1000):
            #     mean_actions_logits = torch.zeros((1, self.action_space.n), device=mean_actions.device)
            #     for i in range(self.action_space.n):
            #         mean_actions_by_first_action = self.alpha * mean_actions[torch.nonzero(first_action[:] == i)[:, 0], :]
            #         mean_actions_logits[0, i] = th.logsumexp(mean_actions_by_first_action.flatten(), dim=0, keepdim=False)
            # print("Time of 1000x2: ", time.time() - t2)
            # t3 = time.time()
            # for t in range(1000):
            #     squash_q = th.sum(th.exp(self.alpha * mean_actions), dim=1, keepdim=True)
            #     M = torch.zeros(self.action_space.n, first_action.shape[0], device=mean_actions.device)
            #     M[first_action, torch.arange(first_action.shape[0]).reshape(-1, 1)] = 1
            #     mean_actions_logits = torch.log(torch.mm(M, squash_q)).transpose(1, 0)
            # print("Time of 1000x3: ", time.time() - t3)
            # t4 = time.time()
            # for t in range(1000):
            # squash_q = th.sum(th.exp(self.alpha * mean_actions), dim=1, keepdim=True)
            # mean_actions_logits = torch.log(torch.zeros(self.action_space.n, 1, device=squash_q.device).scatter_add(0, first_action.to(squash_q.device), squash_q)).transpose(1, 0)
            # print("Time of 1000x4: ", time.time() - t4)
        depth0_logits = self.compute_value(leaves_obs=obs)[0] if self.learn_alpha else th.tensor(0)
        if th.any(th.isnan(mean_actions_logits)):
            import pdb
            pdb.set_trace()
            print("NaN in forward:mean_actions_logits!!!")
            mean_actions_logits[th.isnan(mean_actions_logits)] = 0
        if th.any(th.isnan(depth0_logits)):
            import pdb
            pdb.set_trace()
            print("NaN in forward:depth0_logits!!!")
            depth0_logits[th.isnan(depth0_logits)] = 0
        mean_actions_logits = self.alpha * mean_actions_logits + (1 - self.alpha) * depth0_logits
        mean_actions_logits = add_regularization_logits(mean_actions_logits, self.regularization)
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
        self.add_gradients_history()
        batch_size = obs.shape[0]
        mean_actions_logits = torch.zeros((batch_size, self.action_space.n), device=actions.device)
        ret_values = torch.zeros((batch_size, 1), device=actions.device)
        # Preprocess the observation if needed
        hash_obses = self.hash_obs(obs)
        all_leaves_obs = [] if self.use_leaves_v else [obs]
        all_rewards = []
        all_first_actions = []
        for i in range(batch_size):
            hash_obs = hash_obses[i].item()
            if hash_obs in self.obs2leaves_dict:
                leaves_observations, rewards, first_action = self.obs2leaves_dict.get(hash_obs)
            else:
                print("This shouldn't happen! observation not in our dictionary")
                leaves_observations, rewards, first_action = self.cule_bfs.bfs(obs, self.cule_bfs.max_depth)
                self.obs2leaves_dict[hash_obs] = leaves_observations, rewards, first_action
            all_leaves_obs.append(leaves_observations)
            all_rewards.append(rewards)
            all_first_actions.append(first_action)
            # Preprocess the observation if needed
        all_rewards_th = th.cat(all_rewards).reshape([-1, 1])
        val_coef = self.cule_bfs.gamma ** self.cule_bfs.max_depth
        cat_features = self.extract_features(th.cat(all_leaves_obs))
        shared_features = self.mlp_extractor.shared_net(cat_features)
        if self.use_leaves_v:
            latent_pi = self.mlp_extractor.policy_net(shared_features)
            latent_vf_root = self.mlp_extractor.value_net(shared_features)
        else:
            latent_pi = self.mlp_extractor.policy_net(shared_features[batch_size:])
            latent_vf_root = self.mlp_extractor.value_net(shared_features[:batch_size])
        values = self.value_net(latent_vf_root)
        # Assaf added
        mean_actions = val_coef * self.action_net(latent_pi) + all_rewards_th
        if self.use_leaves_v:
            values_subtrees = val_coef * values + all_rewards_th
        subtree_width = self.action_space.n ** self.cule_bfs.max_depth
        if self.cule_bfs.max_width != -1:
            subtree_width = min(subtree_width, self.cule_bfs.max_width*self.action_space.n)
        # TODO: Optimize here
        for i in range(batch_size):
            mean_actions_batch = mean_actions[subtree_width * i:subtree_width * (i + 1)]
            if self.use_leaves_v:
                ret_values[i, 0] = values_subtrees[subtree_width * i:subtree_width * (i + 1)].max()
            if self.cule_bfs.max_width == -1:
                subtree_width = self.action_space.n ** self.cule_bfs.max_depth
                mean_actions_per_subtree = self.beta * mean_actions_batch.reshape([self.action_space.n, -1])
                counts = th.ones([1, self.action_space.n]) * mean_actions_per_subtree.shape[1]
            else:
                mean_actions_per_subtree = th.zeros(self.action_space.n, mean_actions_batch.shape[0], mean_actions_batch.shape[1],
                                                    device=mean_actions_batch.device) # - 1e6
                idxes = th.arange(mean_actions_batch.shape[0])
                counts = th.zeros(self.action_space.n)
                v, c = th.unique(all_first_actions[i], return_counts=True)
                counts[v] = c.type(th.float32) * self.action_space.n
                mean_actions_per_subtree[all_first_actions[i].flatten(), idxes, :] = mean_actions_batch
                mean_actions_per_subtree = self.beta * mean_actions_per_subtree.reshape([self.action_space.n, -1])
            counts = counts.to(mean_actions.device).reshape([1, -1])
            if self.is_cumulative_mode:
                mean_actions_logits[i, :] = th.sum(mean_actions_per_subtree, dim=1, keepdim=True).transpose(1, 0) / counts
            else:
                mean_actions_logits[i, :] = th.logsumexp(mean_actions_per_subtree, dim=1, keepdim=True).transpose(1, 0) - \
                                  th.log(counts)
            mean_actions_logits[i, counts[0, :] == 0] = -math.inf

        depth0_logits = self.compute_value(leaves_obs=obs)[0] if self.learn_alpha else th.tensor(0)
        if th.any(th.isnan(mean_actions_logits)):
            import pdb
            pdb.set_trace()
            print("NaN in eval_actions:mean_actions_logits!!!")
            mean_actions_logits[th.isnan(mean_actions_logits)] = 0
        if th.any(th.isnan(depth0_logits)):
            import pdb
            pdb.set_trace()
            print("NaN in eval_actions:depth0_logits!!!")
            depth0_logits[th.isnan(depth0_logits)] = 0

        mean_actions_logits = self.alpha * mean_actions_logits + (1 - self.alpha) * depth0_logits
        mean_actions_logits = add_regularization_logits(mean_actions_logits, self.regularization)
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions_logits)
        log_prob = distribution.log_prob(actions)
        # old_values, old_log_probs, old_dist = super(ActorCriticCnnPolicy, self).evaluate_actions(obs, actions)
        # print("Bla", old_values - values, old_log_probs - log_prob, old_dist - distribution.entropy())
        if self.use_leaves_v:
            return ret_values, log_prob, distribution.entropy()
        else:
            return values, log_prob, distribution.entropy()

    def hash_obs(self, obs):
        # return (obs[:, -2:, :, :].double() + obs[:, -2:, :, :].double().pow(2)).view(obs.shape[0], -1).sum(dim=1).int()
        return (obs[:, -2:, :, :].int()).view(obs.shape[0], -1).sum(dim=1)

    def compute_value2(self, leaves_obs, root_obs=None):
        if root_obs is None:
            shared_features = self.mlp_extractor.shared_net(self.extract_features(leaves_obs))
            return self.action_net(self.mlp_extractor.policy_net(shared_features)), None
        cat_features = self.extract_features(th.cat((root_obs, leaves_obs)))
        shared_features = self.mlp_extractor.shared_net(cat_features)
        latent_pi = self.mlp_extractor.policy_net(shared_features[1:])
        latent_vf_root = self.mlp_extractor.value_net(shared_features[:1])
        value_root = self.value_net(latent_vf_root)
        return latent_pi, value_root

    def compute_value(self, leaves_obs, root_obs=None):
        if root_obs is None:
            shared_features = self.mlp_extractor.shared_net(self.extract_features(leaves_obs))
            return self.action_net(self.mlp_extractor.policy_net(shared_features)), None
        shared_features = self.mlp_extractor.shared_net(self.extract_features(leaves_obs))
        latent_pi = self.mlp_extractor.policy_net(shared_features)
        latent_vf_root = self.mlp_extractor.value_net(shared_features)
        value_root = self.value_net(latent_vf_root)
        return latent_pi, value_root