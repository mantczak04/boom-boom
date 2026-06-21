import numpy as np

from prob_minesweeper.env import ProbMinesweeperEnv
from prob_minesweeper.wrappers import FlattenObservationWrapper


def test_flattened_observation_shape_dtype_and_space():
    env = FlattenObservationWrapper(
        ProbMinesweeperEnv(width=3, height=2, obs_mode="state+prob")
    )
    try:
        obs, _ = env.reset(seed=7)
        assert obs.shape == (2 * 3 * 4,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_step_works_through_flatten_wrapper():
    env = FlattenObservationWrapper(
        ProbMinesweeperEnv(width=2, height=2, obs_mode="state+prob")
    )
    try:
        _, info = env.reset(seed=3)
        obs, reward, terminated, truncated, next_info = env.step(
            int(np.flatnonzero(info["action_mask"])[0])
        )
        assert obs.shape == (16,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert next_info["action_mask"].shape == (4,)
    finally:
        env.close()
