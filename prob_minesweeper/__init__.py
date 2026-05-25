"""Probabilistic Minesweeper — Gymnasium-compatible RL environment."""

from gymnasium.envs.registration import register

from prob_minesweeper.board import Board, Cell, RevealResult
from prob_minesweeper.distributions import (
    ConstantDistribution,
    CorrelatedDistribution,
    MineDistribution,
    UniformDistribution,
    make_distribution,
)
from prob_minesweeper.env import ProbMinesweeperEnv
from prob_minesweeper.rewards import RewardConfig

__version__ = "0.1.0"

register(
    id="ProbMinesweeper-v0",
    entry_point="prob_minesweeper.env:ProbMinesweeperEnv",
)

__all__ = [
    "__version__",
    "Board",
    "Cell",
    "RevealResult",
    "RewardConfig",
    "ProbMinesweeperEnv",
    "MineDistribution",
    "CorrelatedDistribution",
    "UniformDistribution",
    "ConstantDistribution",
    "make_distribution",
]
