"""Probabilistic Minesweeper — Gymnasium-compatible RL environment."""

from gymnasium.envs.registration import register

from prob_minesweeper.board import CLUE_MODES, Board, Cell, RevealResult
from prob_minesweeper.distributions import (
    ConstantDistribution,
    CorrelatedDistribution,
    MineDistribution,
    UniformDistribution,
    make_distribution,
)
from prob_minesweeper.env import INITIAL_REVEAL_MODES, ProbMinesweeperEnv
from prob_minesweeper.rewards import REWARD_MODES, RewardConfig, make_reward_config

__version__ = "0.1.0"

register(
    id="ProbMinesweeper-v0",
    entry_point="prob_minesweeper.env:ProbMinesweeperEnv",
)

__all__ = [
    "__version__",
    "Board",
    "CLUE_MODES",
    "Cell",
    "RevealResult",
    "RewardConfig",
    "REWARD_MODES",
    "make_reward_config",
    "ProbMinesweeperEnv",
    "INITIAL_REVEAL_MODES",
    "MineDistribution",
    "CorrelatedDistribution",
    "UniformDistribution",
    "ConstantDistribution",
    "make_distribution",
]
