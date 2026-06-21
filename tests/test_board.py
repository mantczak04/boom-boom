"""Tests for board state logic."""

import numpy as np
import pytest

from prob_minesweeper.board import Board, RevealResult
from prob_minesweeper.distributions import ConstantDistribution


def _board_with_p(p: float, height: int = 3, width: int = 3) -> Board:
    p_mines = ConstantDistribution(p=p).generate(height, width, np.random.default_rng(0))
    return Board.create(height, width, p_mines)


def test_neighbour_p_sum() -> None:
    p_mines = np.array(
        [
            [0.1, 0.2, 0.0],
            [0.3, 0.5, 0.4],
            [0.0, 0.1, 0.2],
        ],
        dtype=np.float32,
    )
    # eight neighbours of center (1, 1), excluding the center itself
    expected = 0.1 + 0.2 + 0.0 + 0.3 + 0.4 + 0.0 + 0.1 + 0.2
    assert Board.neighbour_p_sum(p_mines, 1, 1) == round(expected, 1)


def test_reveal_safe_sets_display_value() -> None:
    board = _board_with_p(0.0)
    board.new_episode(np.random.default_rng(0))
    result = board.reveal(1, 1)
    assert result == RevealResult.SAFE
    cell = board.cell(1, 1)
    assert cell.has_mine is False
    assert cell.display_value == Board.neighbour_p_sum(board.p_mine_field(), 1, 1)


def test_actual_count_uses_sampled_neighbouring_mines() -> None:
    p_mines = np.full((3, 3), 0.25, dtype=np.float32)
    board = Board.create(3, 3, p_mines, clue_mode="actual_count")
    board.new_episode(np.random.default_rng(1))

    assert board.neighbour_mine_count(1, 1) == 1
    assert board.reveal(1, 1) == RevealResult.SAFE
    assert board.cell(1, 1).display_value == 1.0


def test_prob_sum_remains_available_as_default_clue() -> None:
    p_mines = np.full((3, 3), 0.25, dtype=np.float32)
    board = Board.create(3, 3, p_mines)
    board.new_episode(np.random.default_rng(1))

    board.reveal(1, 1)
    assert board.cell(1, 1).display_value == 2.0


def test_neighbour_mine_count_requires_episode() -> None:
    board = _board_with_p(0.0)
    with pytest.raises(RuntimeError, match="new_episode"):
        board.neighbour_mine_count(1, 1)


def test_invalid_clue_mode_raises() -> None:
    with pytest.raises(ValueError, match="clue_mode"):
        Board.create(
            2,
            2,
            np.zeros((2, 2), dtype=np.float32),
            clue_mode="unknown",
        )


def test_reveal_noop_on_already_revealed() -> None:
    board = _board_with_p(0.0)
    board.new_episode(np.random.default_rng(0))
    board.reveal(0, 0)
    assert board.reveal(0, 0) == RevealResult.NOOP


def test_reveal_mine_is_loss() -> None:
    board = _board_with_p(1.0)
    board.new_episode(np.random.default_rng(0))
    result = board.reveal(0, 0)
    assert result == RevealResult.MINE_HIT
    assert board.is_loss()
    assert not board.is_win()


def test_win_all_safe_cells_revealed() -> None:
    board = Board.create(2, 2, np.zeros((2, 2), dtype=np.float32))
    board.new_episode(np.random.default_rng(0))
    for index in range(4):
        row, col = board.from_flat_index(index)
        board.reveal(row, col)
    assert board.is_win()
    assert not board.is_loss()


def test_win_allows_hidden_mines() -> None:
    p_mines = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    board = Board.create(2, 2, p_mines)
    rng = np.random.default_rng(0)
    board.new_episode(rng)
    assert board.hidden_mine_mask().tolist() == [[False, True], [False, True]]
    board.reveal(0, 0)
    board.reveal(1, 0)
    assert board.is_win()
    assert not board.cell(0, 1).is_revealed


def test_new_episode_reproducible_with_seed() -> None:
    p_mines = np.full((3, 3), 0.5, dtype=np.float32)
    board_a = Board.create(3, 3, p_mines)
    board_b = Board.create(3, 3, p_mines)
    board_a.new_episode(np.random.default_rng(42))
    board_b.new_episode(np.random.default_rng(42))
    np.testing.assert_array_equal(board_a.hidden_mine_mask(), board_b.hidden_mine_mask())


def test_new_episode_can_force_cells_safe() -> None:
    board = _board_with_p(1.0)
    safe_cells = ((0, 0), (0, 1), (1, 0), (1, 1))
    board.new_episode(np.random.default_rng(0), guaranteed_safe=safe_cells)

    for row, col in safe_cells:
        assert not board.hidden_mine_mask()[row, col]
        assert not board.cell(row, col).is_revealed


def test_guaranteed_safe_cell_must_be_in_bounds() -> None:
    board = _board_with_p(0.5)
    with pytest.raises(ValueError, match="outside"):
        board.new_episode(np.random.default_rng(0), guaranteed_safe=((3, 0),))


def test_reveal_before_new_episode_raises() -> None:
    board = _board_with_p(0.0)
    with pytest.raises(RuntimeError, match="new_episode"):
        board.reveal(0, 0)


def test_flat_index_roundtrip() -> None:
    board = _board_with_p(0.2, height=4, width=5)
    for index in range(20):
        row, col = board.from_flat_index(index)
        assert board.flat_index(row, col) == index
