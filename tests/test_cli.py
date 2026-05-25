"""Tests for CLI helpers and commands."""

import pytest

from prob_minesweeper.cli import (
    _build_parser,
    format_benchmark_stats,
    main,
    parse_action,
    run_benchmark,
)
from prob_minesweeper.cli import BenchmarkStats
from prob_minesweeper.distributions import ConstantDistribution


def test_parse_action_flat_index() -> None:
    assert parse_action("4", height=3, width=3) == 4


def test_parse_action_row_col() -> None:
    assert parse_action("1 2", height=3, width=3) == 5


def test_parse_action_quit() -> None:
    assert parse_action("q", height=3, width=3) is None
    assert parse_action("quit", height=3, width=3) is None


def test_parse_action_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="outside"):
        parse_action("9 0", height=3, width=3)


def test_parse_action_invalid_format() -> None:
    with pytest.raises(ValueError, match="flat index"):
        parse_action("1 2 3", height=3, width=3)


def test_run_benchmark_outcomes_sum_to_episodes() -> None:
    stats = run_benchmark(
        episodes=20,
        width=2,
        height=2,
        distribution="constant",
        seed=0,
    )
    assert stats.episodes == 20
    assert stats.wins + stats.losses + stats.truncated == 20


def test_run_benchmark_all_safe_wins() -> None:
    stats = run_benchmark(
        episodes=10,
        width=2,
        height=2,
        distribution=ConstantDistribution(p=0.0),
        seed=1,
    )
    assert stats.wins == 10
    assert stats.losses == 0
    assert stats.truncated == 0


def test_run_benchmark_all_mines_lose() -> None:
    stats = run_benchmark(
        episodes=10,
        width=2,
        height=2,
        distribution=ConstantDistribution(p=1.0),
        seed=1,
    )
    assert stats.losses == 10
    assert stats.wins == 0


def test_format_benchmark_stats() -> None:
    text = format_benchmark_stats(
        BenchmarkStats(
            episodes=10,
            wins=3,
            losses=5,
            truncated=2,
            total_reward=1.5,
            total_steps=40,
        )
    )
    assert "Episodes: 10" in text
    assert "Wins: 3 (30.0%)" in text
    assert "Mean reward: 0.150" in text


def test_run_benchmark_invalid_episodes() -> None:
    with pytest.raises(ValueError, match="episodes"):
        run_benchmark(episodes=0)


def test_main_benchmark(capsys) -> None:
    main(
        [
            "benchmark",
            "--episodes",
            "3",
            "--width",
            "2",
            "--height",
            "2",
            "--distribution",
            "constant",
            "--p",
            "0",
            "--seed",
            "1",
        ]
    )
    out = capsys.readouterr().out
    assert "Episodes: 3" in out
    assert "Wins: 3 (100.0%)" in out


def test_build_parser_requires_command() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])
