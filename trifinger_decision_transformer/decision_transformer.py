from time import time
import gc

import gymnasium
import numpy as np
import torch

import d3rlpy
from trifinger_rl_datasets import PolicyBase, PolicyConfig
import trifinger_rl_datasets

from . import policies


class DecisionTransformerBasePolicy(PolicyBase):

    def __init__(
        self,
        model_path,
        action_space,
        observation_space,
        episode_length,
    ):
        self.action_space = action_space
        self.device = "cpu" # "cuda:0"
        self.dtype = np.float32

        self.model = d3rlpy.load_learnable(model_path, device=self.device)
        self.algorithm = self.model.as_stateful_wrapper(target_return=750)
        self.counter = 0
        # to be set in derived classes
        self.dummy_env = None
        self.achieved_position_slice = None
        self.achieved_orientation_slice = None
        self.achieved_keypoints_slice = None
        self.desired_position_slice = None
        self.desired_orientation_slice = None
        self.desired_keypoints_slice = None

    @staticmethod
    def get_policy_config():
        return PolicyConfig(
            flatten_obs=True,
            image_obs=False,
        )

    def reset(self):
        self.counter = 0
        self.algorithm.reset()
        gc.collect()

    def _get_slice(self, observation, slice_tuple):
        if slice_tuple is None:
            return None
        else:
            return observation[slice(*slice_tuple)]

    def get_action(self, observation):
        time0 = time()
        achieved_goal = {
            "object_position": self._get_slice(observation, self.achieved_position_slice),
            "object_orientation": self._get_slice(observation, self.achieved_orientation_slice),
            "object_keypoints": self._get_slice(observation, self.achieved_keypoints_slice),
        }
        desired_goal = {
            "object_position": self._get_slice(observation, self.desired_position_slice),
            "object_orientation": self._get_slice(observation, self.desired_orientation_slice),
            "object_keypoints": self._get_slice(observation, self.desired_keypoints_slice),
        }
        reward = self.dummy_env.compute_reward(achieved_goal, desired_goal, None)
        action = self.algorithm.predict(observation, reward)
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        self.counter += 1
        if self.counter % 750 == 0: # Reset for DT, if the time sequence is too long, it will exceed the embeeding range.
            self.algorithm.reset()
        print(f'prediction_time: {time()-time0}')
        return action


class DTPushPolicy(DecisionTransformerBasePolicy):
    """Decision transformer policy for the push task.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        # model = policies.get_model_path("push_123.d3")
        # model = policies.get_model_path("push_233.d3")
        model = policies.get_model_path("push_888.d3")
        super().__init__(model, action_space, observation_space, episode_length)
        self.dummy_env = gymnasium.make("trifinger-cube-push-real-expert-v0")
        self.obs_indices, self.obs_shapes = self.dummy_env.get_obs_indices()
        print(self.obs_indices)
        self.achieved_position_slice = self.obs_indices["achieved_goal"]["object_position"]
        # self.achieved_orientation_slice = self.obs_indices["achieved_goal"]["object_orientation"]
        self.desired_position_slice = self.obs_indices["desired_goal"]["object_position"]
        # self.desired_orientation_slice = self.obs_indices["desired_goal"]["object_orientation"]


class DTLiftPolicy(DecisionTransformerBasePolicy):
    """Decision transformer policy for the lift task.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        # model = policies.get_model_path("lift_123.d3")
        # model = policies.get_model_path("lift_238.d3")
        model = policies.get_model_path("lift_857.d3")
        super().__init__(model, action_space, observation_space, episode_length)
        self.dummy_env = gymnasium.make("trifinger-cube-lift-real-expert-v0")
        self.obs_indices, self.obs_shapes = self.dummy_env.get_obs_indices()
        print(self.obs_indices)
        self.achieved_keypoints_slice = self.obs_indices["achieved_goal"]["object_keypoints"]
        # self.achieved_orientation_slice = self.obs_indices["achieved_goal"]["object_orientation"]
        self.desired_keypoints_slice = self.obs_indices["desired_goal"]["object_keypoints"]
        # self.desired_orientation_slice = self.obs_indices["desired_goal"]["object_orientation"]
