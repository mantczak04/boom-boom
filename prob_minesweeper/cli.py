"""CLI entry points for prob-minesweeper."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from prob_minesweeper.distributions import MineDistribution
from prob_minesweeper.env import ProbMinesweeperEnv


@dataclass(frozen=True)
class BenchmarkStats:
    """Aggregate results from a benchmark run."""

    episodes: int
    wins: int
    losses: int
    truncated: int
    total_reward: float
    total_steps: int

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.episodes if self.episodes else 0.0

    @property
    def mean_steps(self) -> float:
        return self.total_steps / self.episodes if self.episodes else 0.0


def parse_action(text: str, height: int, width: int) -> int | None:
    """Parse user input into a flat cell index, or ``None`` to quit."""
    stripped = text.strip().lower()
    if stripped in ("q", "quit", "exit"):
        return None

    parts = stripped.split()
    if len(parts) == 1:
        row, col = divmod(int(parts[0]), width)
    elif len(parts) == 2:
        row, col = int(parts[0]), int(parts[1])
    else:
        raise ValueError("Enter a flat index, or 'row col', or 'q' to quit")

    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(f"Cell ({row}, {col}) is outside the {height}x{width} board")
    return row * width + col


def run_benchmark(
    *,
    episodes: int,
    width: int = 9,
    height: int = 9,
    distribution: str | MineDistribution = "correlated",
    distribution_kwargs: dict[str, float] | None = None,
    seed: int | None = None,
) -> BenchmarkStats:
    """Run a random valid-action agent for ``episodes`` episodes."""
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
        for episode in range(episodes):
            _, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            episode_reward = 0.0
            steps = 0
            terminated = truncated_flag = False

            while not (terminated or truncated_flag):
                mask = info["action_mask"]
                valid = np.flatnonzero(mask)
                if len(valid) == 0:
                    break
                action = int(rng.choice(valid))
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


def format_benchmark_stats(stats: BenchmarkStats) -> str:
    """Return a human-readable benchmark summary."""
    n = stats.episodes
    pct = lambda count: 100.0 * count / n if n else 0.0
    lines = [
        f"Episodes: {n}",
        f"Wins: {stats.wins} ({pct(stats.wins):.1f}%)",
        f"Losses: {stats.losses} ({pct(stats.losses):.1f}%)",
        f"Truncated: {stats.truncated} ({pct(stats.truncated):.1f}%)",
        f"Mean reward: {stats.mean_reward:.3f}",
        f"Mean steps: {stats.mean_steps:.1f}",
    ]
    return "\n".join(lines)


def cmd_play(args: argparse.Namespace) -> None:
    """Interactive human play loop."""
    env = ProbMinesweeperEnv(
        width=args.width,
        height=args.height,
        distribution=args.distribution,
        obs_mode="state+prob",
        render_mode="human",
        seed=args.seed,
    )
    try:
        _, info = env.reset(seed=args.seed)
        env.render()
        print(
            "Reveal a cell: flat index, or 'row col' (0-based). "
            "Type 'q' to quit."
        )

        while True:
            if not info["action_mask"].any():
                break
            try:
                raw = input("> ").strip()
            except EOFError:
                print()
                break

            try:
                action = parse_action(raw, env.height, env.width)
            except ValueError as exc:
                print(exc)
                continue

            if action is None:
                break
            if not info["action_mask"][action]:
                print("That cell is already revealed.")
                continue

            _, reward, terminated, truncated, info = env.step(action)
            env.render()
            print(f"reward={reward:+.2f}")

            if terminated:
                if env.board and env.board.is_win():
                    print("You win!")
                else:
                    print("Mine hit — game over.")
                break
            if truncated:
                print("Step limit reached.")
                break
    finally:
        env.close()


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Print random-agent benchmark statistics."""
    distribution_kwargs = None
    if args.distribution == "constant" and args.p is not None:
        distribution_kwargs = {"p": args.p}

    stats = run_benchmark(
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        distribution=args.distribution,
        distribution_kwargs=distribution_kwargs,
        seed=args.seed,
    )
    print(format_benchmark_stats(stats))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob-minesweeper",
        description="Probabilistic Minesweeper — play or benchmark",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_env_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--width", type=int, default=9, help="Board width")
        p.add_argument("--height", type=int, default=9, help="Board height")
        p.add_argument(
            "--distribution",
            default="correlated",
            choices=("correlated", "uniform", "constant"),
            help="Mine probability field generator",
        )
        p.add_argument("--seed", type=int, default=None, help="RNG seed")

    play = sub.add_parser("play", help="Interactive human play")
    add_env_args(play)
    play.set_defaults(func=cmd_play)

    bench = sub.add_parser("benchmark", help="Random-agent benchmark")
    add_env_args(bench)
    bench.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes to run",
    )
    bench.add_argument(
        "--p",
        type=float,
        default=None,
        help="Mine probability for constant distribution",
    )
    bench.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
