"""ASCII and rgb_array rendering for ProbMinesweeperEnv."""

from __future__ import annotations

import numpy as np

from prob_minesweeper.board import Board, Cell

CELL_PX = 16

_GRID_COLOR = (48, 48, 48)
_UNREVEALED_COLOR = (70, 70, 90)
_MINE_COLOR = (220, 50, 50)
_EMPTY_COLOR = (245, 245, 245)
_MAX_CLUE = 8.0


def _cell_symbol(cell: Cell) -> str:
    """Single-cell ASCII symbol per AGENTS.md legend."""
    if not cell.is_revealed:
        return "#"
    if cell.has_mine:
        return "X"
    if cell.display_value == 0.0:
        return "."
    return f"{cell.display_value:.1f}"


def _clue_color(display_value: float) -> tuple[int, int, int]:
    """Blue intensity scales with neighbour p-sum (classic clue analogue)."""
    t = min(1.0, display_value / _MAX_CLUE)
    red = int(250 - 90 * t)
    green = int(230 - 120 * t)
    blue = int(160 + 95 * t)
    return (red, green, blue)


def _draw_grid(
    image: np.ndarray, height: int, width: int, cell_px: int
) -> None:
    """Draw 1px separators between cells."""
    for row in range(1, height):
        y = row * cell_px - 1
        image[y, :, :] = _GRID_COLOR
    for col in range(1, width):
        x = col * cell_px - 1
        image[:, x, :] = _GRID_COLOR


def ascii_render(board: Board) -> str:
    """Return an ASCII grid (legend: . / clue / # / X)."""
    symbols = [
        [_cell_symbol(board.cell(row, col)) for col in range(board.width)]
        for row in range(board.height)
    ]
    cell_width = max(len(symbol) for row in symbols for symbol in row)
    lines = [
        " ".join(symbol.rjust(cell_width) for symbol in row) for row in symbols
    ]
    return "\n".join(lines)


def to_rgb_array(board: Board, *, cell_px: int = CELL_PX) -> np.ndarray:
    """Return an RGB image of shape ``(H * cell_px, W * cell_px, 3)``."""
    if cell_px < 1:
        raise ValueError(f"cell_px must be >= 1, got {cell_px}")

    height_px = board.height * cell_px
    width_px = board.width * cell_px
    image = np.zeros((height_px, width_px, 3), dtype=np.uint8)

    for row in range(board.height):
        for col in range(board.width):
            cell = board.cell(row, col)
            if not cell.is_revealed:
                color = _UNREVEALED_COLOR
            elif cell.has_mine:
                color = _MINE_COLOR
            elif cell.display_value == 0.0:
                color = _EMPTY_COLOR
            else:
                color = _clue_color(cell.display_value)

            y0, y1 = row * cell_px, (row + 1) * cell_px
            x0, x1 = col * cell_px, (col + 1) * cell_px
            image[y0:y1, x0:x1] = color

    _draw_grid(image, board.height, board.width, cell_px)
    return image


__all__ = ["CELL_PX", "ascii_render", "to_rgb_array"]
