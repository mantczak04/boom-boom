"""Rainbow env adapter and network shape tests."""

from __future__ import annotations

from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")

from rainbow.env_adapter import Env
from rainbow.model import DQN

OBS_CHANNELS = {"state": 3, "state+prob": 4}


def obs_channels_for_mode(obs_mode: str) -> int:
    return OBS_CHANNELS[obs_mode]


def make_args(**overrides) -> Namespace:
    defaults = {
        "board_width": 9,
        "board_height": 9,
        "obs_mode": "state+prob",
        "obs_channels": 4,
        "atoms": 51,
        "hidden_size": 256,
        "noisy_std": 0.1,
        "device": "cpu",
        "seed": 0,
        "distribution": "correlated",
        "distribution_kwargs": {},
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_obs_channels_for_both_modes() -> None:
    assert obs_channels_for_mode("state") == 3
    assert obs_channels_for_mode("state+prob") == 4


@pytest.mark.parametrize(
    ("obs_mode", "obs_channels"),
    [("state+prob", 4), ("state", 3)],
)
def test_env_reset_mask_and_forward_pass(obs_mode: str, obs_channels: int) -> None:
    args = make_args(
        obs_mode=obs_mode,
        obs_channels=obs_channels_for_mode(obs_mode),
    )
    action_space = args.board_width * args.board_height

    env = Env(args)
    try:
        state = env.reset()
        assert state.shape == (obs_channels, args.board_height, args.board_width)
        assert state.dtype == torch.float32

        mask = env.action_mask()
        assert mask.shape == (action_space,)
        assert mask.dtype == torch.bool

        net = DQN(args, action_space=action_space)
        net.eval()
        with torch.no_grad():
            output = net(state.unsqueeze(0))
        assert output.shape == (1, action_space, args.atoms)
    finally:
        env.close()


def test_env_step_returns_state_reward_done() -> None:
    args = make_args()
    env = Env(args)
    try:
        env.reset()
        action = int(env.action_mask().nonzero(as_tuple=True)[0][0].item())
        state, reward, done = env.step(action)
        assert state.shape == (4, 9, 9)
        assert state.dtype == torch.float32
        assert isinstance(reward, float)
        assert isinstance(done, bool)
    finally:
        env.close()
