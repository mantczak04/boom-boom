"""Compare available agents on the same probabilistic Minesweeper configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.common import add_distribution_args, build_distribution_kwargs
from prob_minesweeper.agents import DQNAgent, MaskablePPOAgent, MinRiskAgent, RandomAgent
from prob_minesweeper.evaluation import compare_agents
from prob_minesweeper.rewards import REWARD_MODES, make_reward_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
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
    parser.add_argument(
        "--dqn-model",
        type=Path,
        default=Path("models/dqn_prob_minesweeper.zip"),
    )
    parser.add_argument(
        "--maskable-ppo-model",
        type=Path,
        default=Path("models/maskable_ppo_prob_minesweeper.zip"),
    )
    return parser


def display_agent_name(agent_name: str) -> str:
    if agent_name == "Min-risk":
        return "Min-risk (oracle)"
    return agent_name


def print_table(results: list[Any]) -> None:
    headers = [
        "Agent",
        "Episodes",
        "Wins",
        "Losses",
        "Truncated",
        "Win rate",
        "Mean reward",
        "Mean steps",
    ]
    rows = [
        [
            display_agent_name(r.agent_name),
            str(r.episodes),
            str(r.wins),
            str(r.losses),
            str(r.truncated),
            f"{r.win_rate:.2%}",
            f"{r.mean_reward:.3f}",
            f"{r.mean_steps:.2f}",
        ]
        for r in results
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def print_agent_diagnostics(agents: list[Any]) -> None:
    for agent in agents:
        invalid_action_rate = getattr(agent, "invalid_action_rate", None)
        if invalid_action_rate is not None:
            print(f"{agent.name} invalid action rate: {invalid_action_rate:.2%}")


def main() -> None:
    args = build_parser().parse_args()
    distribution_kwargs = build_distribution_kwargs(args)

    agents: list[Any] = [
        RandomAgent(args.seed),
        MinRiskAgent(),
    ]

    if args.dqn_model.is_file():
        try:
            dqn_agent = DQNAgent(
                args.dqn_model,
                fallback_mode="random",
                seed=args.seed,
            )
            dqn_agent.name = "DQN (random fallback)"
            agents.append(dqn_agent)
        except ImportError as exc:
            print(f"Skipping DQN: {exc}")
    else:
        print(f"Skipping DQN: model not found at {args.dqn_model}")

    if args.maskable_ppo_model.is_file():
        try:
            agents.append(MaskablePPOAgent(args.maskable_ppo_model))
        except ImportError as exc:
            print(f"Skipping MaskablePPO: {exc}")
    else:
        print(f"Skipping MaskablePPO: model not found at {args.maskable_ppo_model}")

    results = compare_agents(
        agents,
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

    print_table(results)
    print()
    print_agent_diagnostics(agents)


if __name__ == "__main__":
    main()
