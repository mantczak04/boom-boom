"""Train a Stable-Baselines3 DQN on probabilistic Minesweeper."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from prob_minesweeper.env import ProbMinesweeperEnv
from prob_minesweeper.wrappers import FlattenObservationWrapper


def make_dqn_env(
    *,
    width: int = 5,
    height: int = 5,
    distribution: str = "correlated",
    seed: int | None = None,
) -> FlattenObservationWrapper:
    env = ProbMinesweeperEnv(
        width=width,
        height=height,
        distribution=distribution,
        obs_mode="state+prob",
        render_mode=None,
        seed=seed,
    )
    return FlattenObservationWrapper(env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument(
        "--distribution",
        choices=("constant", "uniform", "correlated"),
        default="correlated",
    )
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("models/dqn_prob_minesweeper.zip")
    )
    return parser


def train(args: argparse.Namespace) -> Path:
    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise SystemExit(
            "Stable-Baselines3 is required. Run `uv sync --dev --extra rl`."
        ) from exc

    if args.timesteps < 1:
        raise ValueError("timesteps must be >= 1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = make_dqn_env(
        width=args.width,
        height=args.height,
        distribution=args.distribution,
        seed=args.seed,
    )
    model: Any = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.95,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=500,
        train_freq=4,
        device="cpu",
        verbose=1,
        seed=args.seed,
    )
    try:
        model.learn(total_timesteps=args.timesteps)
        model.save(args.output)
    finally:
        env.close()
    return args.output


def main() -> None:
    args = build_parser().parse_args()
    output = train(args)
    print(f"Saved DQN model to {output}")


if __name__ == "__main__":
    main()
