"""Rainbow replay memory batch shape test."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tests.test_rainbow import make_args
from rainbow.memory import ReplayMemory


def test_replay_memory_sample_shape() -> None:
    args = make_args()
    args.history_length = 1
    args.discount = 0.99
    args.multi_step = 3
    args.priority_weight = 0.4
    args.priority_exponent = 0.5
    args.batch_size = 4

    mem = ReplayMemory(args, capacity=512)
    for step in range(200):
        state = torch.randn(args.obs_channels, args.board_height, args.board_width)
        terminal = step > 0 and step % 25 == 0
        mem.append(state, step % 81, float(step), terminal)

    _, states, _, _, next_states, _, _ = mem.sample(args.batch_size)
    assert states.shape == (args.batch_size, args.obs_channels, args.board_height, args.board_width)
    assert next_states.shape == (args.batch_size, args.obs_channels, args.board_height, args.board_width)
