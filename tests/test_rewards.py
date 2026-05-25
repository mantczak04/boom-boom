"""Tests for reward configuration."""

import pytest

from prob_minesweeper.board import RevealResult
from prob_minesweeper.rewards import RewardConfig


@pytest.mark.parametrize(
    ("factory", "p_mine", "expected_safe", "expected_mine"),
    [
        (RewardConfig.risk_adjusted, 0.2, 0.8, -0.2),
        (RewardConfig.risk_adjusted, 0.0, 1.0, 0.0),
        (RewardConfig.risk_adjusted, 1.0, 0.0, -1.0),
        (RewardConfig.sparse, 0.5, 0.0, -1.0),
        (RewardConfig.uniform, 0.5, 1.0, -1.0),
    ],
)
def test_factory_reveal_and_mine_rewards(
    factory, p_mine, expected_safe, expected_mine
) -> None:
    config = factory()
    assert config.reveal_reward_fn(p_mine) == pytest.approx(expected_safe)
    assert config.mine_penalty_fn(p_mine) == pytest.approx(expected_mine)


@pytest.mark.parametrize("factory", [RewardConfig.risk_adjusted, RewardConfig.sparse, RewardConfig.uniform])
def test_win_bonus(factory) -> None:
    assert factory().win_bonus == 1.0


def test_reward_for_reveal_noop() -> None:
    config = RewardConfig.risk_adjusted()
    assert config.reward_for_reveal(0.5, RevealResult.NOOP) == 0.0


def test_reward_for_reveal_safe_and_win() -> None:
    config = RewardConfig.risk_adjusted()
    assert config.reward_for_reveal(0.2, RevealResult.SAFE) == pytest.approx(0.8)
    assert config.reward_for_reveal(0.2, RevealResult.WIN) == pytest.approx(1.8)


def test_reward_for_reveal_mine_hit() -> None:
    config = RewardConfig.uniform()
    assert config.reward_for_reveal(0.9, RevealResult.MINE_HIT) == -1.0


def test_sparse_ignores_p_mine_on_safe_reveal() -> None:
    config = RewardConfig.sparse()
    for p in (0.0, 0.5, 1.0):
        assert config.reward_for_reveal(p, RevealResult.SAFE) == 0.0
        assert config.reward_for_reveal(p, RevealResult.WIN) == pytest.approx(1.0)
