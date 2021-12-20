from typing import Tuple
import torch as th
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from cule_bfs import CuleBFS


class ActorCriticCnnTSPolicy(ActorCriticCnnPolicy):
    def __init__(self, **kwargs):
        super(ActorCriticCnnPolicy, self).__init__(kwargs)
        self.cule_bfs = CuleBFS(env_name=full_env_name, tree_depth=args.tree_depth, verbose=False,
                                ale_start_steps=1, ignore_value_function=False, perturb_reward=True, step_env=env.env,
                                args=args)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        leaves_observations = self.cule_bfs.bfs(obs, )
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob
