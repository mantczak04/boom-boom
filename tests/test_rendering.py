"""Tests for ASCII and rgb_array rendering."""

import numpy as np
import pytest

from prob_minesweeper.board import Board
from prob_minesweeper.distributions import ConstantDistribution
from prob_minesweeper.env import ProbMinesweeperEnv
from prob_minesweeper.rendering import CELL_PX, ascii_render, to_rgb_array


def _board(p: float, height: int = 3, width: int = 3) -> Board:
    p_mines = ConstantDistribution(p=p).generate(
        height, width, np.random.default_rng(0)
    )
    board = Board.create(height, width, p_mines)
    board.new_episode(np.random.default_rng(0))
    return board


def test_ascii_legend_symbols() -> None:
    board = _board(0.0)
    assert ascii_render(board) == "# # #\n# # #\n# # #"

    board.reveal(1, 1)
    rendered = ascii_render(board)
    assert "." in rendered
    assert "#" in rendered

    mine_board = _board(1.0)
    mine_board.reveal(0, 0)
    assert "X" in ascii_render(mine_board)


def test_ascii_formats_clue_one_decimal() -> None:
    p_mines = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    board = Board.create(3, 3, p_mines)
    board.new_episode(np.random.default_rng(0))
    board.reveal(1, 1)
    assert board.cell(1, 1).display_value == 1.5
    assert "1.5" in ascii_render(board)


def test_rgb_array_shape_and_dtype() -> None:
    board = _board(0.0, height=2, width=3)
    image = to_rgb_array(board)
    assert image.shape == (2 * CELL_PX, 3 * CELL_PX, 3)
    assert image.dtype == np.uint8


def test_rgb_array_custom_cell_px() -> None:
    board = _board(0.0, height=2, width=2)
    image = to_rgb_array(board, cell_px=8)
    assert image.shape == (16, 16, 3)


def test_rgb_array_invalid_cell_px() -> None:
    with pytest.raises(ValueError, match="cell_px"):
        to_rgb_array(_board(0.0), cell_px=0)


def test_rgb_unrevealed_differs_from_empty() -> None:
    board = _board(0.0)
    image = to_rgb_array(board)
    unrevealed = tuple(image[0, 0])
    board.reveal(0, 0)
    empty = tuple(to_rgb_array(board)[0, 0])
    assert unrevealed != empty


def test_rgb_mine_cell_is_red() -> None:
    board = _board(1.0)
    board.reveal(0, 0)
    pixel = tuple(to_rgb_array(board)[0, 0])
    assert pixel[0] > pixel[1]


def test_env_render_rgb_array() -> None:
    env = ProbMinesweeperEnv(width=3, height=2, render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None
    assert frame.shape == (2 * CELL_PX, 3 * CELL_PX, 3)


def test_env_render_human(capsys) -> None:
    env = ProbMinesweeperEnv(width=2, height=2, render_mode="human")
    env.reset(seed=0)
    assert env.render() is None
    out = capsys.readouterr().out
    assert "#" in out
