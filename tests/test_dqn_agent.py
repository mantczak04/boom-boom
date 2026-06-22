from pathlib import Path

import numpy as np
import pytest

stable_baselines3 = pytest.importorskip("stable_baselines3")

from prob_minesweeper.agents.dqn_agent import DQNAgent
from prob_minesweeper.distributions import ConstantDistribution
from prob_minesweeper.env import ProbMinesweeperEnv


class FakeModel:
    def __init__(self, action: int) -> None:
        self.action = action
        self.last_observation = None

    def predict(self, observation, deterministic=True):
        assert deterministic is True
        self.last_observation = observation
        return np.array([self.action]), None


def make_agent(monkeypatch, tmp_path: Path, model: FakeModel) -> DQNAgent:
    model_path = tmp_path / "model.zip"
    model_path.touch()
    monkeypatch.setattr(
        stable_baselines3.DQN, "load", lambda path, device="auto": model
    )
    return DQNAgent(model_path)


def test_missing_model_path_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="DQN model not found"):
        DQNAgent(tmp_path / "missing.zip")


def test_invalid_prediction_uses_valid_fallback(monkeypatch, tmp_path):
    model = FakeModel(action=0)
    agent = make_agent(monkeypatch, tmp_path, model)
    env = ProbMinesweeperEnv(
        width=2,
        height=1,
        distribution=ConstantDistribution(0.0),
        obs_mode="state+prob",
    )
    try:
        obs, _ = env.reset(seed=1)
        obs, _, _, _, info = env.step(0)
        action = agent.select_action(obs, info, env)
        assert action == 1
        assert info["action_mask"][action]
    finally:
        env.close()


def test_invalid_prediction_increments_invalid_action_counter(monkeypatch, tmp_path):
    model = FakeModel(action=0)
    agent = make_agent(monkeypatch, tmp_path, model)

    env = ProbMinesweeperEnv(
        width=2,
        height=1,
        distribution=ConstantDistribution(0.0),
        obs_mode="state+prob",
    )
    try:
        obs, _ = env.reset(seed=1)
        obs, _, _, _, info = env.step(0)

        action = agent.select_action(obs, info, env)

        assert action == 1
        assert agent.predictions == 1
        assert agent.invalid_predictions == 1
        assert agent.invalid_action_rate == pytest.approx(1.0)
    finally:
        env.close()


def test_valid_prediction_does_not_increment_invalid_counter(monkeypatch, tmp_path):
    model = FakeModel(action=0)
    agent = make_agent(monkeypatch, tmp_path, model)

    env = ProbMinesweeperEnv(
        width=2,
        height=1,
        distribution=ConstantDistribution(0.0),
        obs_mode="state+prob",
    )
    try:
        obs, info = env.reset(seed=1)

        action = agent.select_action(obs, info, env)

        assert action == 0
        assert agent.predictions == 1
        assert agent.invalid_predictions == 0
        assert agent.invalid_action_rate == pytest.approx(0.0)
    finally:
        env.close()


def test_observation_is_flattened_before_prediction(monkeypatch, tmp_path):
    model = FakeModel(action=0)
    agent = make_agent(monkeypatch, tmp_path, model)
    env = ProbMinesweeperEnv(width=3, height=2, obs_mode="state+prob")
    try:
        obs, info = env.reset(seed=2)
        agent.select_action(obs, info, env)
        assert model.last_observation.shape == (1, 2 * 3 * 4)
        assert model.last_observation.dtype == np.float32
    finally:
        env.close()
