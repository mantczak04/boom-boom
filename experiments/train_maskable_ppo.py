"""Train sb3-contrib MaskablePPO on probabilistic Minesweeper."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import add_distribution_args, build_distribution_kwargs
from prob_minesweeper.env import ProbMinesweeperEnv
from prob_minesweeper.rewards import REWARD_MODES, make_reward_config
from prob_minesweeper.wrappers import FlattenObservationWrapper


def make_maskable_ppo_env(
    *,
    width: int = 5,
    height: int = 5,
    distribution: str = "correlated",
    distribution_kwargs: dict[str, Any] | None = None,
    obs_mode: str = "state",
    clue_mode: str = "actual_count",
    initial_reveal: str = "safe_2x2",
    reward_mode: str = "completion",
    seed: int | None = None,
) -> FlattenObservationWrapper:
    env = ProbMinesweeperEnv(
        width=width,
        height=height,
        distribution=distribution,
        distribution_kwargs=distribution_kwargs,
        obs_mode=obs_mode,
        clue_mode=clue_mode,
        initial_reveal=initial_reveal,
        reward_config=make_reward_config(reward_mode),
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
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/maskable_ppo_prob_minesweeper.zip"),
    )
    return parser


def train(args: argparse.Namespace) -> Path:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit(
            "sb3-contrib is required. Run `uv sync --dev --extra rl`."
        ) from exc

    if args.timesteps < 1:
        raise ValueError("timesteps must be >= 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    distribution_kwargs = build_distribution_kwargs(args)

    env = make_maskable_ppo_env(
        width=args.width,
        height=args.height,
        distribution=args.distribution,
        distribution_kwargs=distribution_kwargs,
        obs_mode=args.obs_mode,
        clue_mode=args.clue_mode,
        initial_reveal=args.initial_reveal,
        reward_mode=args.reward_mode,
        seed=args.seed,
    )

    mask = env.action_masks()
    if mask.shape != (args.width * args.height,):
        raise RuntimeError(f"Invalid action mask shape: {mask.shape}")
    if mask.dtype != np.bool_:
        raise RuntimeError(f"Invalid action mask dtype: {mask.dtype}")

    model: Any = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        device="cpu",
        verbose=args.verbose,
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
    print(f"Saved MaskablePPO model to {output}")


if __name__ == "__main__":
    main()
