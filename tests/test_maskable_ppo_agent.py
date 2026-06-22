import numpy as np
import pytest

sb3_contrib = pytest.importorskip("sb3_contrib")

from prob_minesweeper.agents.maskable_ppo_agent import MaskablePPOAgent


class FakeMaskablePPOModel:
    def __init__(self, action: int) -> None:
        self.action = action
        self.last_observation = None
        self.last_action_masks = None
        self.observation_space = None

    def predict(self, observation, deterministic=True, action_masks=None):
        assert deterministic is True
        self.last_observation = observation
        self.last_action_masks = action_masks
        return np.array([self.action]), None


def test_missing_maskable_ppo_model_path_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="MaskablePPO model not found"):
        MaskablePPOAgent(tmp_path / "missing.zip")


def test_maskable_ppo_agent_passes_flattened_obs_and_mask(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.touch()

    fake_model = FakeMaskablePPOModel(action=1)

    monkeypatch.setattr(
        sb3_contrib.MaskablePPO,
        "load",
        lambda path, device="auto": fake_model,
    )

    agent = MaskablePPOAgent(model_path)

    obs = np.zeros((2, 2, 3), dtype=np.float32)
    info = {"action_mask": np.array([False, True, False, True])}

    action = agent.select_action(obs, info, env=None)

    assert action == 1
    assert fake_model.last_observation.shape == (1, 2 * 2 * 3)
    assert fake_model.last_observation.dtype == np.float32
    np.testing.assert_array_equal(
        fake_model.last_action_masks,
        info["action_mask"],
    )


def test_maskable_ppo_invalid_prediction_raises(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.touch()

    fake_model = FakeMaskablePPOModel(action=0)

    monkeypatch.setattr(
        sb3_contrib.MaskablePPO,
        "load",
        lambda path, device="auto": fake_model,
    )

    agent = MaskablePPOAgent(model_path)
    obs = np.zeros((2, 2, 3), dtype=np.float32)
    info = {"action_mask": np.array([False, True, False, True])}

    with pytest.raises(RuntimeError, match="predicted invalid action"):
        agent.select_action(obs, info, env=None)
