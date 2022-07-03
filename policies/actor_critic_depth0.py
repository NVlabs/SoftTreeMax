from typing import Tuple

import torch
import torch as th
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.distributions import Distribution
from policies.cule_bfs import CuleBFS


class ActorCriticCnnPolicyDepth0(ActorCriticCnnPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(ActorCriticCnnPolicyDepth0, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.gradients_history = {}

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
        return super(ActorCriticCnnPolicyDepth0, self).evaluate_actions(obs, actions)

    def add_gradients_history(self):
        policy_params = {param_name: param for param_name, param in self.named_parameters() if
                         param_name.startswith('action_net')}
        for param_name in policy_params:
            if policy_params[param_name].grad is None:
                break
            if param_name not in self.gradients_history:
                self.gradients_history[param_name] = []
            self.gradients_history[param_name].append(policy_params[param_name].grad.data.detach().clone())
