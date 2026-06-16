# -*- coding: utf-8 -*-
# Adapted from Kaixhin/Rainbow env.py (MIT).
# Replaced the Atari ALE wrapper with ProbMinesweeperEnv.
#
# State contract: (C, H, W) float32 tensor; history_length = 1 (no frame stacking).
# The only place (H, W, C) gym observations are transposed to channels-first.

from __future__ import annotations

import torch

from prob_minesweeper.env import ProbMinesweeperEnv


class Env:
    """Gymnasium ProbMinesweeper wrapper returning PyTorch (C, H, W) states."""

    def __init__(self, args) -> None:
        self.device = args.device
        self.board_width = args.board_width
        self.board_height = args.board_height
        self.obs_mode = getattr(args, "obs_mode", "state+prob")
        self.distribution = getattr(args, "distribution", "correlated")
        self.distribution_kwargs = getattr(args, "distribution_kwargs", None) or {}
        self.training = True

        self._env = ProbMinesweeperEnv(
            width=self.board_width,
            height=self.board_height,
            distribution=self.distribution,
            distribution_kwargs=self.distribution_kwargs,
            obs_mode=self.obs_mode,
            render_mode=None,
        )
        self._action_mask: torch.Tensor | None = None

    def _to_tensor(self, obs) -> torch.Tensor:
        # Gymnasium returns (H, W, C); PyTorch convs expect (C, H, W).
        return torch.tensor(obs, dtype=torch.float32, device=self.device).permute(2, 0, 1)

    def reset(self) -> torch.Tensor:
        obs, info = self._env.reset()
        self._action_mask = torch.tensor(
            info["action_mask"], dtype=torch.bool, device=self.device
        )
        return self._to_tensor(obs)

    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._action_mask = torch.tensor(
            info["action_mask"], dtype=torch.bool, device=self.device
        )
        done = bool(terminated or truncated)
        return self._to_tensor(obs), float(reward), done

    def action_mask(self) -> torch.Tensor:
        if self._action_mask is None:
            raise RuntimeError("Call reset() before action_mask()")
        return self._action_mask

    def action_space(self) -> int:
        return self.board_height * self.board_width

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def close(self) -> None:
        self._env.close()
