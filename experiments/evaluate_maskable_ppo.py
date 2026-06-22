"""Evaluate a saved MaskablePPO model without retraining it."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common import add_distribution_args, build_distribution_kwargs
from prob_minesweeper.agents import MaskablePPOAgent
from prob_minesweeper.evaluation import EvaluationResult, evaluate_agent
from prob_minesweeper.rewards import REWARD_MODES, make_reward_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/maskable_ppo_prob_minesweeper.zip"),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument(
        "--distribution",
        choices=("constant", "uniform", "correlated"),
        default="correlated",
    )
    add_distribution_args(parser)
    parser.add_argument(
        "--obs-mode", choices=("state", "state+prob"), default="state"
    )
    parser.add_argument(
        "--clue-mode", choices=("prob_sum", "actual_count"), default="actual_count"
    )
    parser.add_argument(
        "--initial-reveal", choices=("none", "safe_2x2"), default="safe_2x2"
    )
    parser.add_argument(
        "--reward-mode", choices=REWARD_MODES, default="completion"
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
    distribution_kwargs = build_distribution_kwargs(args)

    print("Environment configuration")
    configuration = (
        ("width", args.width),
        ("height", args.height),
        ("distribution", args.distribution),
        ("distribution_kwargs", distribution_kwargs),
        ("obs_mode", args.obs_mode),
        ("clue_mode", args.clue_mode),
        ("initial_reveal", args.initial_reveal),
        ("reward_mode", args.reward_mode),
    )
    label_width = max(len(label) for label, _ in configuration)
    for label, value in configuration:
        print(f"{label:<{label_width}} : {value}")
    print()

    agent = MaskablePPOAgent(args.model)
    result = evaluate_agent(
        agent,
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        distribution=args.distribution,
        distribution_kwargs=distribution_kwargs,
        obs_mode=args.obs_mode,
        clue_mode=args.clue_mode,
        initial_reveal=args.initial_reveal,
        reward_config=make_reward_config(args.reward_mode),
        seed=args.seed,
    )
    print_result(result)


if __name__ == "__main__":
    main()
