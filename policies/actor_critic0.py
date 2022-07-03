from typing import Tuple

import torch
import torch as th
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.distributions import Distribution
from policies.cule_bfs import CuleBFS


class ActorCriticCnn0(ActorCriticCnnPolicy):
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        policy_params = {param_name: param for param_name, param in self.named_parameters() if param_name.startswith('action_net')}
        for param_name in policy_params:
            if policy_params[param_name].grad is None:
                break
            if not param_name in self.policy_gradients:
                self.policy_gradients[param_name] = []
            self.policy_gradients[param_name].append(policy_params[param_name].grad.data.clone().detach())
        return super(ActorCriticCnn0, self).evaluate_actions(obs, actions)
