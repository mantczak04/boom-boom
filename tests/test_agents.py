import numpy as np
import pytest

from prob_minesweeper.agents import MinRiskAgent, RandomAgent
from prob_minesweeper.distributions import ConstantDistribution
from prob_minesweeper.env import ProbMinesweeperEnv


def test_random_agent_only_selects_valid_actions():
    info = {"action_mask": np.array([False, True, False, True])}
    agent = RandomAgent(seed=3)
    assert {agent.select_action(np.array([]), info, None) for _ in range(30)} <= {1, 3}


def test_min_risk_selects_lowest_valid_probability_and_ignores_revealed():
    env = ProbMinesweeperEnv(width=2, height=1, distribution=ConstantDistribution(0.0))
    obs, info = env.reset(seed=1)
    env.board.cell(0, 0).p_mine = 0.1
    env.board.cell(0, 1).p_mine = 0.7
    assert MinRiskAgent().select_action(obs, info, env) == 0
    obs, _, _, _, info = env.step(0)
    assert MinRiskAgent().select_action(obs, info, env) == 1


@pytest.mark.parametrize("agent", [RandomAgent(1), MinRiskAgent()])
def test_agents_fail_clearly_without_valid_actions(agent):
    with pytest.raises(RuntimeError, match="No valid actions"):
        agent.select_action(np.array([]), {"action_mask": np.zeros(1, dtype=bool)}, None)
