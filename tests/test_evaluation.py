import pytest

from prob_minesweeper.agents import MinRiskAgent, RandomAgent
from prob_minesweeper.distributions import ConstantDistribution
from prob_minesweeper.evaluation import EvaluationResult, evaluate_agent
from prob_minesweeper.rewards import RewardConfig


@pytest.mark.parametrize("agent", [RandomAgent(2), MinRiskAgent()])
def test_zero_probability_boards_are_always_won(agent):
    result = evaluate_agent(
        agent,
        episodes=4,
        width=2,
        height=2,
        distribution=ConstantDistribution(0.0),
        initial_reveal="none",
        seed=9,
    )
    assert result.wins == 4
    assert result.losses == result.truncated == 0


def test_one_probability_boards_are_always_lost():
    result = evaluate_agent(
        MinRiskAgent(), episodes=3, width=2, height=2,
        distribution=ConstantDistribution(1.0), initial_reveal="none", seed=4
    )
    assert result.losses == 3
    assert result.wins + result.losses + result.truncated == result.episodes


def test_result_properties():
    result = EvaluationResult("x", 4, 1, 2, 1, 6.0, 10)
    assert result.win_rate == 0.25
    assert result.loss_rate == 0.5
    assert result.mean_reward == 1.5
    assert result.mean_steps == 2.5


def test_safe_opening_is_not_rewarded_or_counted_as_agent_steps():
    result = evaluate_agent(
        RandomAgent(2),
        episodes=1,
        width=3,
        height=3,
        distribution=ConstantDistribution(0.0),
        seed=9,
    )
    assert result.wins == 1
    assert result.total_steps == 5
    assert result.total_reward == pytest.approx(10.5)


def test_evaluation_accepts_original_rule_variant():
    result = evaluate_agent(
        RandomAgent(1),
        episodes=2,
        width=2,
        height=2,
        distribution=ConstantDistribution(0.0),
        obs_mode="state+prob",
        clue_mode="prob_sum",
        initial_reveal="none",
        reward_config=RewardConfig.risk_adjusted(),
        seed=3,
    )
    assert result.wins == 2
