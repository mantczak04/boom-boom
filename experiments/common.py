"""Shared helpers for experiment scripts."""

from __future__ import annotations

import argparse
from typing import Any


def add_distribution_args(parser: argparse.ArgumentParser) -> None:
    """Add optional distribution parameter arguments to an experiment parser."""
    parser.add_argument(
        "--p",
        type=float,
        default=None,
        help="Mine probability for constant distribution",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=None,
        help="Lower bound for uniform distribution",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=None,
        help="Upper bound for uniform distribution",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Gaussian smoothing sigma for correlated distribution",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Scale parameter for correlated distribution",
    )


def build_distribution_kwargs(args: argparse.Namespace) -> dict[str, Any] | None:
    """Build ProbMinesweeperEnv distribution_kwargs from parsed CLI args."""
    if args.distribution == "constant":
        return {"p": args.p} if args.p is not None else None

    if args.distribution == "uniform":
        kwargs: dict[str, Any] = {}
        if args.low is not None:
            kwargs["low"] = args.low
        if args.high is not None:
            kwargs["high"] = args.high
        return kwargs or None

    if args.distribution == "correlated":
        kwargs: dict[str, Any] = {}
        if args.sigma is not None:
            kwargs["sigma"] = args.sigma
        if args.scale is not None:
            kwargs["scale"] = args.scale
        return kwargs or None

    return None
