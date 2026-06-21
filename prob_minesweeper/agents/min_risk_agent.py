"""Minimum mine-risk baseline with privileged probability access."""

from typing import Any

import numpy as np


class MinRiskAgent:
    """Use hidden ``p_mine`` values; an oracle in hidden-risk mode."""

    name = "Min-risk"

    def select_action(self, obs: np.ndarray, info: dict[str, Any], env: Any) -> int:
        del obs
        valid = np.flatnonzero(info["action_mask"])
        if len(valid) == 0:
            raise RuntimeError("No valid actions available")
        if env.board is None:
            raise RuntimeError("Environment has not been reset")
        return int(
            min(
                valid,
                key=lambda action: env.board.cell(
                    *env.board.from_flat_index(int(action))
                ).p_mine,
            )
        )


__all__ = ["MinRiskAgent"]
