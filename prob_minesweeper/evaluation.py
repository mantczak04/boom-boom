"""Reusable agent evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from prob_minesweeper.env import ProbMinesweeperEnv


@dataclass(frozen=True)
class EvaluationResult:
    agent_name: str
    episodes: int
    wins: int
    losses: int
    truncated: int
    total_reward: float
    total_steps: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes if self.episodes else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.episodes if self.episodes else 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.episodes if self.episodes else 0.0

    @property
    def mean_steps(self) -> float:
        return self.total_steps / self.episodes if self.episodes else 0.0


def evaluate_agent(
    agent: Any,
    *,
    episodes: int,
    width: int = 5,
    height: int = 5,
    distribution: Any = "correlated",
    distribution_kwargs: dict[str, Any] | None = None,
    obs_mode: str = "state+prob",
    seed: int | None = None,
) -> EvaluationResult:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    env = ProbMinesweeperEnv(
        width=width,
        height=height,
        distribution=distribution,
        distribution_kwargs=distribution_kwargs,
        obs_mode=obs_mode,
    )
    wins = losses = truncations = total_steps = 0
    total_reward = 0.0
    try:
        for episode in range(episodes):
            episode_seed = None if seed is None else seed + episode
            obs, info = env.reset(seed=episode_seed)
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.select_action(obs, info, env)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                total_steps += 1
            if truncated:
                truncations += 1
            elif env.board is not None and env.board.is_win():
                wins += 1
            else:
                losses += 1
    finally:
        env.close()
    return EvaluationResult(
        agent_name=agent.name,
        episodes=episodes,
        wins=wins,
        losses=losses,
        truncated=truncations,
        total_reward=total_reward,
        total_steps=total_steps,
    )


def compare_agents(
    agents: Iterable[Any],
    *,
    episodes: int,
    width: int,
    height: int,
    distribution: Any,
    distribution_kwargs: dict[str, Any] | None = None,
    seed: int | None = None,
) -> list[EvaluationResult]:
    return [
        evaluate_agent(
            agent,
            episodes=episodes,
            width=width,
            height=height,
            distribution=distribution,
            distribution_kwargs=distribution_kwargs,
            seed=seed,
        )
        for agent in agents
    ]


__all__ = ["EvaluationResult", "compare_agents", "evaluate_agent"]
