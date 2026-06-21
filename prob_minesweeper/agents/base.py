"""Agent interface used by demos and evaluation."""

from typing import Any, Protocol

import numpy as np


class Agent(Protocol):
    name: str

    def select_action(self, obs: np.ndarray, info: dict[str, Any], env: Any) -> int:
        """Return one valid flat cell index."""


__all__ = ["Agent"]
