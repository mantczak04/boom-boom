"""Observation wrappers used by reinforcement-learning models."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlattenObservationWrapper(gym.ObservationWrapper):
    """Flatten an ``H x W x C`` board observation for an MLP policy."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("FlattenObservationWrapper requires a Box observation space")
        flat_size = int(np.prod(env.observation_space.shape))
        self.observation_space = spaces.Box(
            low=float(np.min(env.observation_space.low)),
            high=float(np.max(env.observation_space.high)),
            shape=(flat_size,),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32).reshape(-1)


__all__ = ["FlattenObservationWrapper"]
