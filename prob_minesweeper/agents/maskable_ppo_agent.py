"""sb3-contrib MaskablePPO adapter for the common agent interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class MaskablePPOAgent:
    """Load a trained MaskablePPO model and select valid masked actions."""

    name = "MaskablePPO"

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"MaskablePPO model not found: {path}")

        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise ImportError(
                "MaskablePPOAgent requires sb3-contrib. Install it with "
                "`uv sync --dev --extra rl`."
            ) from exc

        self.model = MaskablePPO.load(path, device="cpu")

    def select_action(
        self, obs: np.ndarray, info: dict[str, Any], env: Any
    ) -> int:
        del env

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
                "MaskablePPO model observation shape does not match this board: "
                f"expected {tuple(expected_shape)}, got {tuple(flat_obs.shape[1:])}. "
                "Train and evaluate MaskablePPO with the same width, height, "
                "and obs_mode."
            )

        predicted, _ = self.model.predict(
            flat_obs,
            deterministic=True,
            action_masks=mask,
        )
        action = int(np.asarray(predicted).reshape(-1)[0])

        if 0 <= action < len(mask) and mask[action]:
            return action

        raise RuntimeError(
            f"MaskablePPO predicted invalid action {action} despite action mask"
        )


__all__ = ["MaskablePPOAgent"]
