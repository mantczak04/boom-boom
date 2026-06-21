"""Stable-Baselines3 DQN adapter for the common agent interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from prob_minesweeper.agents.min_risk_agent import MinRiskAgent


class DQNAgent:
    """Load a trained DQN and ensure that it returns a legal board action."""

    name = "DQN"

    def __init__(self, model_path: str | Path, fallback_agent: Any | None = None) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"DQN model not found: {path}")

        try:
            from stable_baselines3 import DQN
        except ImportError as exc:
            raise ImportError(
                "DQNAgent requires Stable-Baselines3. Install it with "
                "`uv sync --dev --extra rl`."
            ) from exc

        self.model = DQN.load(path, device="cpu")
        self.fallback_agent = (
            fallback_agent if fallback_agent is not None else MinRiskAgent()
        )
        self._rng = np.random.default_rng()

    def select_action(
        self, obs: np.ndarray, info: dict[str, Any], env: Any
    ) -> int:
        mask = np.asarray(info["action_mask"], dtype=np.bool_).reshape(-1)
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            raise RuntimeError("No valid actions available")

        flat_obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        model_space = getattr(self.model, "observation_space", None)
        expected_shape = getattr(model_space, "shape", None)
        if expected_shape is not None and tuple(expected_shape) != tuple(
            flat_obs.shape[1:]
        ):
            raise ValueError(
                "DQN model observation shape does not match this board: "
                f"expected {tuple(expected_shape)}, got {tuple(flat_obs.shape[1:])}. "
                "Train and evaluate DQN with the same width, height, and obs_mode."
            )
        predicted, _ = self.model.predict(flat_obs, deterministic=True)
        action = int(np.asarray(predicted).reshape(-1)[0])
        if 0 <= action < len(mask) and mask[action]:
            return action

        try:
            fallback_action = int(self.fallback_agent.select_action(obs, info, env))
            if 0 <= fallback_action < len(mask) and mask[fallback_action]:
                return fallback_action
        except (IndexError, KeyError, RuntimeError, TypeError, ValueError):
            pass

        return int(self._rng.choice(valid))


__all__ = ["DQNAgent"]
