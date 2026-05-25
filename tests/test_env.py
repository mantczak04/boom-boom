"""Tests for ProbMinesweeperEnv (Gymnasium API + step-9 checklist)."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import prob_minesweeper  # noqa: F401 — registers ProbMinesweeper-v0

from prob_minesweeper.distributions import ConstantDistribution
from prob_minesweeper.env import ProbMinesweeperEnv
from prob_minesweeper.rewards import RewardConfig


def _env(**kwargs) -> ProbMinesweeperEnv:
    kwargs.setdefault("width", 3)
    kwargs.setdefault("height", 3)
    return ProbMinesweeperEnv(**kwargs)


@pytest.mark.parametrize("obs_mode", ["state", "state+prob"])
def test_env_checker(obs_mode: str) -> None:
    """Full Gymnasium API compliance (render + both observation modes)."""
    env = gym.make(
        "ProbMinesweeper-v0",
        width=5,
        height=5,
        obs_mode=obs_mode,
        render_mode="rgb_array",
    )
    try:
        check_env(env.unwrapped, skip_render_check=False)
    finally:
        env.close()


def test_gym_make_registered_env() -> None:
    env = gym.make("ProbMinesweeper-v0", width=4, height=4)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (4, 4, 3)
        assert info["action_mask"].shape == (16,)
        assert env.spec.id == "ProbMinesweeper-v0"
    finally:
        env.close()


def test_observation_channels() -> None:
    state_env = _env(obs_mode="state")
    obs, _ = state_env.reset(seed=0)
    assert obs.shape == (3, 3, 3)
    assert state_env.observation_space.shape == (3, 3, 3)

    prob_env = _env(obs_mode="state+prob")
    obs, _ = prob_env.reset(seed=0)
    assert obs.shape == (3, 3, 4)
    assert prob_env.observation_space.shape == (3, 3, 4)


def test_state_plus_prob_hides_p_on_revealed() -> None:
    env = _env(obs_mode="state+prob", distribution=ConstantDistribution(p=0.7))
    env.reset(seed=1)
    obs, _, _, _, _ = env.step(0)
    row, col = env.board.from_flat_index(0)
    assert env.board.cell(row, col).is_revealed
    assert obs[row, col, 3] == 0.0
    assert obs[row, col, 0] == 1.0


def test_action_mask_masks_revealed_cells() -> None:
    env = _env(distribution=ConstantDistribution(p=0.0))
    _, info = env.reset(seed=0)
    assert info["action_mask"].shape == (9,)
    assert np.all(info["action_mask"])

    env.step(0)
    *_, info = env.step(0)
    assert not info["action_mask"][0]
    assert info["action_mask"].sum() == 8


def test_action_mask_excludes_revealed_through_episode() -> None:
    """No revealed cell may appear as a valid action at any step."""
    env = _env(distribution=ConstantDistribution(p=0.0))
    _, info = env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        revealed = env.board.revealed_mask().reshape(-1)
        assert not np.any(info["action_mask"] & revealed)
        valid = np.flatnonzero(info["action_mask"])
        if len(valid) == 0:
            break
        _, _, terminated, truncated, info = env.step(int(valid[0]))


def test_action_mask_all_false_when_board_full() -> None:
    env = _env(width=2, height=2, distribution=ConstantDistribution(p=0.0))
    env.reset(seed=0)
    for action in range(4):
        env.step(action)
    assert env.board.is_win()
    *_, info = env.step(0)
    assert not info["action_mask"].any()


def test_termination_on_mine_hit() -> None:
    env = _env(distribution=ConstantDistribution(p=1.0))
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated
    assert not truncated
    assert env.board.is_loss()


def test_win_with_hidden_mines_terminates() -> None:
    """Env terminates on win when safe cells are done; mines may stay hidden."""
    from prob_minesweeper.board import Board

    env = _env(width=2, height=2)
    env.reset(seed=0)
    p_field = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    env.board = Board.create(2, 2, p_field)
    env.board.new_episode(np.random.default_rng(0))

    _, _, terminated, _, _ = env.step(env.board.flat_index(0, 0))
    assert not terminated
    _, _, terminated, truncated, _ = env.step(env.board.flat_index(1, 0))
    assert terminated
    assert not truncated
    assert env.board.is_win()
    assert not env.board.cell(0, 1).is_revealed


def test_termination_on_win() -> None:
    env = _env(distribution=ConstantDistribution(p=0.0))
    env.reset(seed=0)
    terminated = False
    for action in range(9):
        _, _, terminated, truncated, _ = env.step(action)
        assert not truncated
    assert terminated
    assert env.board.is_win()


def test_truncation_at_max_steps() -> None:
    env = _env(distribution=ConstantDistribution(p=0.0), max_steps=3)
    env.reset(seed=0)
    env.step(0)
    _, _, terminated, truncated, _ = env.step(1)
    assert not terminated
    assert not truncated
    _, _, terminated, truncated, _ = env.step(2)
    assert truncated
    assert not terminated


@pytest.mark.parametrize(
    ("reward_config", "expected"),
    [
        (RewardConfig.sparse(), 0.0),
        (RewardConfig.uniform(), 1.0),
        (RewardConfig.risk_adjusted(), 1.0),
    ],
)
def test_reward_shape_on_safe_reveal(
    reward_config: RewardConfig, expected: float
) -> None:
    env = _env(
        distribution=ConstantDistribution(p=0.0),
        reward_config=reward_config,
    )
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(0)
    assert reward == pytest.approx(expected)


def test_reward_mine_penalty_sparse_and_uniform() -> None:
    env = _env(
        distribution=ConstantDistribution(p=1.0),
        reward_config=RewardConfig.sparse(),
    )
    env.reset(seed=0)
    _, sparse_reward, _, _, _ = env.step(0)
    assert sparse_reward == -1.0

    env = _env(
        distribution=ConstantDistribution(p=1.0),
        reward_config=RewardConfig.uniform(),
    )
    env.reset(seed=0)
    _, uniform_reward, _, _, _ = env.step(0)
    assert uniform_reward == -1.0


def test_reward_matches_risk_adjusted_config() -> None:
    env = _env(
        distribution=ConstantDistribution(p=0.2),
        reward_config=RewardConfig.risk_adjusted(),
    )
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(0)
    assert reward == pytest.approx(0.8)


def test_reward_noop_on_revealed_cell() -> None:
    env = _env(distribution=ConstantDistribution(p=0.0))
    env.reset(seed=0)
    env.step(0)
    _, reward, terminated, _, _ = env.step(0)
    assert reward == 0.0
    assert not terminated


def test_obs_channel_semantics() -> None:
    env = _env(obs_mode="state", distribution=ConstantDistribution(p=0.25))
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32
    assert np.all(obs[..., 0] == 0.0)
    assert np.all(obs[..., 1] == 0.0)
    assert np.all(obs[..., 2] == 0.0)

    obs, _, _, _, _ = env.step(0)
    row, col = env.board.from_flat_index(0)
    assert obs[row, col, 0] == 1.0
    assert obs[row, col, 1] == pytest.approx(env.board.cell(row, col).display_value)
    assert obs[row, col, 1] >= 0.0


def test_observation_within_space_bounds() -> None:
    env = _env(width=4, height=4, obs_mode="state+prob")
    obs, info = env.reset(seed=1)
    for _ in range(5):
        assert env.observation_space.contains(obs)
        if env.board.is_loss() or env.board.is_win():
            break
        valid = np.flatnonzero(info["action_mask"])
        obs, _, term, trunc, info = env.step(int(valid[0]))
        if term or trunc:
            break
    assert env.observation_space.contains(obs)


def test_reset_seed_reproducibility() -> None:
    env_a = _env(distribution="constant", distribution_kwargs={"p": 0.5})
    env_b = _env(distribution="constant", distribution_kwargs={"p": 0.5})

    obs_a, info_a = env_a.reset(seed=42)
    obs_b, info_b = env_b.reset(seed=42)

    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(info_a["action_mask"], info_b["action_mask"])
    np.testing.assert_array_equal(
        env_a.board.hidden_mine_mask(),
        env_b.board.hidden_mine_mask(),
    )


def test_step_sequence_reproducible_with_seed() -> None:
    env_a = _env(distribution=ConstantDistribution(p=0.5))
    env_b = _env(distribution=ConstantDistribution(p=0.5))
    env_a.reset(seed=7)
    env_b.reset(seed=7)

    rewards_a: list[float] = []
    rewards_b: list[float] = []
    for action in (0, 1, 2, 0):
        _, ra, ta, tra, _ = env_a.step(action)
        _, rb, tb, trb, _ = env_b.step(action)
        rewards_a.append(ra)
        rewards_b.append(rb)
        assert ta == tb and tra == trb

    assert rewards_a == rewards_b


def test_win_reward_includes_bonus() -> None:
    env = _env(
        width=2,
        height=2,
        distribution=ConstantDistribution(p=0.0),
        reward_config=RewardConfig.risk_adjusted(),
    )
    env.reset(seed=0)
    for action in range(3):
        env.step(action)
    _, reward, _, _, _ = env.step(3)
    assert reward == pytest.approx(2.0)


def test_invalid_obs_mode_raises() -> None:
    with pytest.raises(ValueError, match="obs_mode"):
        _env(obs_mode="invalid")


def test_step_before_reset_raises() -> None:
    env = _env()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
