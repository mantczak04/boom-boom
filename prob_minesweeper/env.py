"""ProbMinesweeperEnv — Gymnasium environment implementation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from prob_minesweeper.board import Board, RevealResult
from prob_minesweeper.distributions import MineDistribution, make_distribution
from prob_minesweeper.rewards import RewardConfig

_OBS_MODES = ("state", "state+prob")
_NUM_CHANNELS = {"state": 3, "state+prob": 4}
# Eight neighbours, each p_mine <= 1.0
_MAX_DISPLAY_VALUE = 8.0


class ProbMinesweeperEnv(gym.Env):
    """Probabilistic Minesweeper as a Gymnasium environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        width: int = 9,
        height: int = 9,
        distribution: str | MineDistribution = "correlated",
        distribution_kwargs: dict[str, Any] | None = None,
        obs_mode: str = "state",
        reward_config: RewardConfig | None = None,
        max_steps: int | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if width < 1 or height < 1:
            raise ValueError(f"width and height must be >= 1, got {width}x{height}")
        if obs_mode not in _OBS_MODES:
            valid = ", ".join(_OBS_MODES)
            raise ValueError(f"obs_mode {obs_mode!r} must be one of: {valid}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode {render_mode!r} is not supported")

        self.width = width
        self.height = height
        self.obs_mode = obs_mode
        self._num_channels = _NUM_CHANNELS[obs_mode]
        self._distribution = make_distribution(
            distribution, **(distribution_kwargs or {})
        )
        self.reward_config = reward_config or RewardConfig.risk_adjusted()
        self.max_steps = max_steps if max_steps is not None else height * width * 2
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(height * width)
        self.observation_space = spaces.Box(
            low=0.0,
            high=float(_MAX_DISPLAY_VALUE),
            shape=(height, width, self._num_channels),
            dtype=np.float32,
        )

        self.board: Board | None = None
        self._step_count = 0

        if seed is not None:
            self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)

        p_mines = self._distribution.generate(self.height, self.width, self.np_random)
        self.board = Board.create(self.height, self.width, p_mines)
        self.board.new_episode(self.np_random)
        self._step_count = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.board is None:
            raise RuntimeError("Call reset() before step()")

        row, col = self.board.from_flat_index(int(action))
        cell = self.board.cell(row, col)
        p_mine = cell.p_mine

        result = self.board.reveal(row, col)
        reward = float(self.reward_config.reward_for_reveal(p_mine, result))

        self._step_count += 1
        terminated = self.board.is_loss() or self.board.is_win()
        truncated = not terminated and self._step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        if self.board is None:
            raise RuntimeError("Call reset() before observing")

        obs = np.zeros((self.height, self.width, self._num_channels), dtype=np.float32)
        for row in range(self.height):
            for col in range(self.width):
                cell = self.board.cell(row, col)
                obs[row, col, 0] = float(cell.is_revealed)
                obs[row, col, 1] = float(cell.display_value) if cell.is_revealed else 0.0
                obs[row, col, 2] = 0.0  # is_flagged reserved
                if self._num_channels == 4 and not cell.is_revealed:
                    obs[row, col, 3] = float(cell.p_mine)
        return obs

    def _action_mask(self) -> np.ndarray:
        if self.board is None:
            raise RuntimeError("Call reset() before action_mask()")
        revealed = self.board.revealed_mask().reshape(-1)
        return (~revealed).astype(np.bool_)

    def _get_info(self) -> dict[str, Any]:
        return {"action_mask": self._action_mask()}

    def render(self) -> np.ndarray | None:
        if self.render_mode is None or self.board is None:
            return None

        from prob_minesweeper.rendering import ascii_render, to_rgb_array

        if self.render_mode == "human":
            print(ascii_render(self.board))
            return None
        if self.render_mode == "rgb_array":
            return to_rgb_array(self.board)
        return None

    def close(self) -> None:
        return None


__all__ = ["ProbMinesweeperEnv"]
