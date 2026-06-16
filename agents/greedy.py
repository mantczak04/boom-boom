"""Greedy baseline: reveal the unrevealed cell with the lowest p_mine."""

from __future__ import annotations

import argparse

import numpy as np

from prob_minesweeper.cli import BenchmarkStats, format_benchmark_stats, run_benchmark
from prob_minesweeper.distributions import MineDistribution
from prob_minesweeper.env import ProbMinesweeperEnv


def select_greedy_action(env: ProbMinesweeperEnv, mask: np.ndarray) -> int | None:
    """Pick the masked cell with lowest p_mine; tie-break by lowest flat index."""
    if env.board is None:
        raise RuntimeError("Call reset() before select_greedy_action()")

    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return None

    p_mine = env.board.p_mine_field().ravel()
    return int(valid[np.argmin(p_mine[valid])])


def run_greedy(
    *,
    episodes: int,
    width: int = 9,
    height: int = 9,
    distribution: str | MineDistribution = "correlated",
    distribution_kwargs: dict[str, float] | None = None,
    seed: int | None = None,
) -> BenchmarkStats:
    """Run a greedy p_mine-minimizing agent for ``episodes`` episodes."""
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")

    env = ProbMinesweeperEnv(
        width=width,
        height=height,
        distribution=distribution,
        distribution_kwargs=distribution_kwargs,
        render_mode=None,
    )
    rng = np.random.default_rng(seed)

    wins = losses = truncated = 0
    total_reward = 0.0
    total_steps = 0

    try:
        for _episode in range(episodes):
            _, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            episode_reward = 0.0
            steps = 0
            terminated = truncated_flag = False

            while not (terminated or truncated_flag):
                mask = info["action_mask"]
                action = select_greedy_action(env, mask)
                if action is None:
                    break
                _, reward, terminated, truncated_flag, info = env.step(action)
                episode_reward += float(reward)
                steps += 1

            total_reward += episode_reward
            total_steps += steps
            if terminated and env.board is not None:
                if env.board.is_win():
                    wins += 1
                elif env.board.is_loss():
                    losses += 1
            elif truncated_flag:
                truncated += 1
    finally:
        env.close()

    return BenchmarkStats(
        episodes=episodes,
        wins=wins,
        losses=losses,
        truncated=truncated,
        total_reward=total_reward,
        total_steps=total_steps,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="greedy-benchmark",
        description="Greedy p_mine-minimizing baseline benchmark",
    )
    parser.add_argument("--width", type=int, default=9, help="Board width")
    parser.add_argument("--height", type=int, default=9, help="Board height")
    parser.add_argument(
        "--distribution",
        default="correlated",
        choices=("correlated", "uniform", "constant"),
        help="Mine probability field generator",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=None,
        help="Mine probability for constant distribution",
    )
    parser.add_argument(
        "--compare-random",
        action="store_true",
        help="Also run the random baseline on the same config and print both",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    distribution_kwargs = None
    if args.distribution == "constant" and args.p is not None:
        distribution_kwargs = {"p": args.p}

    common = dict(
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        distribution=args.distribution,
        distribution_kwargs=distribution_kwargs,
        seed=args.seed,
    )

    greedy_stats = run_greedy(**common)
    print("Greedy baseline:")
    print(format_benchmark_stats(greedy_stats))

    if args.compare_random:
        random_stats = run_benchmark(**common)
        print()
        print("Random baseline:")
        print(format_benchmark_stats(random_stats))
        greedy_wr = 100.0 * greedy_stats.wins / greedy_stats.episodes
        random_wr = 100.0 * random_stats.wins / random_stats.episodes
        print()
        print(f"Win-rate: greedy {greedy_wr:.1f}% vs random {random_wr:.1f}%")


if __name__ == "__main__":
    main()
