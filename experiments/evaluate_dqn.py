"""Evaluate a saved DQN model without retraining it."""

from __future__ import annotations

import argparse
from pathlib import Path

from prob_minesweeper.agents import DQNAgent
from prob_minesweeper.evaluation import EvaluationResult, evaluate_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("models/dqn_prob_minesweeper.zip")
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument(
        "--distribution",
        choices=("constant", "uniform", "correlated"),
        default="correlated",
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser


def print_result(result: EvaluationResult) -> None:
    rows = (
        ("Agent", result.agent_name),
        ("Episodes", result.episodes),
        ("Wins", result.wins),
        ("Losses", result.losses),
        ("Truncated", result.truncated),
        ("Win rate", f"{result.win_rate:.2%}"),
        ("Mean reward", f"{result.mean_reward:.3f}"),
        ("Mean steps", f"{result.mean_steps:.2f}"),
    )
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}} : {value}")


def main() -> None:
    args = build_parser().parse_args()
    agent = DQNAgent(args.model)
    result = evaluate_agent(
        agent,
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        distribution=args.distribution,
        seed=args.seed,
    )
    print_result(result)


if __name__ == "__main__":
    main()
