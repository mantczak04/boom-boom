"""Reward configuration for ProbMinesweeperEnv."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from prob_minesweeper.board import RevealResult

REWARD_MODES = ("risk_adjusted", "sparse", "uniform", "completion")


@dataclass(frozen=True)
class RewardConfig:
    """Maps reveal outcomes to scalar rewards."""

    reveal_reward_fn: Callable[[float], float]
    mine_penalty_fn: Callable[[float], float]
    win_bonus: float

    def reward_for_reveal(self, p_mine: float, result: RevealResult) -> float:
        """Return step reward for a reveal attempt (noop → 0)."""
        if result == RevealResult.NOOP:
            return 0.0
        if result == RevealResult.MINE_HIT:
            return self.mine_penalty_fn(p_mine)
        reward = self.reveal_reward_fn(p_mine)
        if result == RevealResult.WIN:
            reward += self.win_bonus
        return reward

    @classmethod
    def risk_adjusted(cls) -> RewardConfig:
        """Default: reward safe reveals by confidence, penalize mines by risk."""
        return cls(
            reveal_reward_fn=lambda p: 1.0 - p,
            mine_penalty_fn=lambda p: -p,
            win_bonus=1.0,
        )

    @classmethod
    def sparse(cls) -> RewardConfig:
        """No shaping on safe reveals; fixed terminal win/loss signals."""
        return cls(
            reveal_reward_fn=lambda _p: 0.0,
            mine_penalty_fn=lambda _p: -1.0,
            win_bonus=1.0,
        )

    @classmethod
    def uniform(cls) -> RewardConfig:
        """Fixed +1 / -1 per reveal regardless of cell probability."""
        return cls(
            reveal_reward_fn=lambda _p: 1.0,
            mine_penalty_fn=lambda _p: -1.0,
            win_bonus=1.0,
        )

    @classmethod
    def completion(cls) -> RewardConfig:
        """Reward board completion without exposing local mine risk."""
        return cls(
            reveal_reward_fn=lambda _p: 0.1,
            mine_penalty_fn=lambda _p: -1.0,
            win_bonus=10.0,
        )


def make_reward_config(mode: str) -> RewardConfig:
    """Build a reward configuration from its CLI/UI name."""
    factories = {
        "risk_adjusted": RewardConfig.risk_adjusted,
        "sparse": RewardConfig.sparse,
        "uniform": RewardConfig.uniform,
        "completion": RewardConfig.completion,
    }
    try:
        return factories[mode]()
    except KeyError as exc:
        valid = ", ".join(REWARD_MODES)
        raise ValueError(f"reward mode {mode!r} must be one of: {valid}") from exc


__all__ = ["REWARD_MODES", "RewardConfig", "make_reward_config"]
