"""Tests for hidden-risk DQN script defaults."""

import pytest

from experiments.evaluate_dqn import build_parser as build_evaluation_parser
from experiments.train_dqn import build_parser as build_training_parser
from experiments.train_dqn import make_dqn_env
from experiments.train_maskable_ppo import (
    build_parser as build_maskable_ppo_training_parser,
)
from experiments.train_maskable_ppo import make_maskable_ppo_env


def test_training_defaults_to_hidden_risk() -> None:
    args = build_training_parser().parse_args([])
    assert args.obs_mode == "state"
    assert args.clue_mode == "actual_count"
    assert args.initial_reveal == "safe_2x2"
    assert args.reward_mode == "completion"
    assert args.timesteps == 500_000


def test_evaluation_defaults_to_hidden_risk() -> None:
    args = build_evaluation_parser().parse_args([])
    assert args.obs_mode == "state"
    assert args.clue_mode == "actual_count"
    assert args.initial_reveal == "safe_2x2"
    assert args.reward_mode == "completion"


def test_dqn_environment_has_visible_state_shape() -> None:
    env = make_dqn_env(
        width=3,
        height=2,
        distribution="constant",
        seed=0,
    )
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (3 * 2 * 3,)
        assert env.unwrapped.obs_mode == "state"
        assert env.unwrapped.clue_mode == "actual_count"
        assert env.unwrapped.initial_reveal == "safe_2x2"
        assert env.unwrapped.reward_config.win_bonus == 10.0
        assert env.unwrapped.board.revealed_mask().sum() == 4
    finally:
        env.close()


def test_evaluation_help_includes_initial_reveal(capsys) -> None:
    parser = build_evaluation_parser()
    assert "--initial-reveal" in parser.format_help()


def test_maskable_ppo_training_defaults_to_hidden_risk() -> None:
    args = build_maskable_ppo_training_parser().parse_args([])
    assert args.obs_mode == "state"
    assert args.clue_mode == "actual_count"
    assert args.initial_reveal == "safe_2x2"
    assert args.reward_mode == "completion"
    assert args.timesteps == 100_000


def test_maskable_ppo_environment_has_visible_state_shape() -> None:
    env = make_maskable_ppo_env(
        width=3,
        height=3,
        distribution="constant",
        distribution_kwargs={"p": 0.15},
        seed=0,
    )
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (3 * 3 * 3,)
        assert env.unwrapped.obs_mode == "state"
        assert env.unwrapped.clue_mode == "actual_count"
        assert env.unwrapped.initial_reveal == "safe_2x2"
        assert env.unwrapped.reward_config.win_bonus == 10.0
        assert env.unwrapped.board.revealed_mask().sum() == 4
        assert env.unwrapped.board.p_mine_field().mean() == pytest.approx(0.15)
    finally:
        env.close()
