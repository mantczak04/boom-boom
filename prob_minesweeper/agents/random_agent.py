"""Uniform random valid-action baseline."""

from typing import Any

import numpy as np


class RandomAgent:
    name = "Random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, info: dict[str, Any], env: Any) -> int:
        del obs, env
        valid = np.flatnonzero(info["action_mask"])
        if len(valid) == 0:
            raise RuntimeError("No valid actions available")
        return int(self.rng.choice(valid))


__all__ = ["RandomAgent"]
