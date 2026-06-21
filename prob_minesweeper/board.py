"""Board state: cells, lazy reveal sampling, clues, win/loss detection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np

_NEIGHBOUR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

CLUE_MODES = ("prob_sum", "actual_count")


@dataclass
class Cell:
    """Single board cell."""

    p_mine: float
    is_revealed: bool = False
    has_mine: bool | None = None
    display_value: float = 0.0


class RevealResult(Enum):
    """Outcome of a reveal attempt."""

    NOOP = "noop"
    SAFE = "safe"
    MINE_HIT = "mine_hit"
    WIN = "win"


@dataclass
class Board:
    """Probabilistic minesweeper board state.

    Hidden mine outcomes are sampled from ``Bernoulli(p_mine)`` at episode start.
    A reveal discloses the cached outcome for that cell (the agent does not see it
    until then). Win when every non-mine cell is revealed; mines may stay hidden.
    """

    height: int
    width: int
    cells: list[list[Cell]]
    clue_mode: str = "prob_sum"
    _mine_outcomes: np.ndarray | None = None

    @classmethod
    def create(
        cls,
        height: int,
        width: int,
        p_mines: np.ndarray,
        clue_mode: str = "prob_sum",
    ) -> Board:
        if p_mines.shape != (height, width):
            raise ValueError(
                f"p_mines shape {p_mines.shape} does not match board ({height}, {width})"
            )
        if clue_mode not in CLUE_MODES:
            valid = ", ".join(CLUE_MODES)
            raise ValueError(f"clue_mode {clue_mode!r} must be one of: {valid}")
        cells = [
            [Cell(p_mine=float(p_mines[row, col])) for col in range(width)]
            for row in range(height)
        ]
        return cls(height=height, width=width, cells=cells, clue_mode=clue_mode)

    def new_episode(
        self,
        rng: np.random.Generator,
        guaranteed_safe: Iterable[tuple[int, int]] = (),
    ) -> None:
        """Sample hidden outcomes, optionally forcing selected cells safe."""
        safe_cells = tuple(guaranteed_safe)
        for row, col in safe_cells:
            if not self.in_bounds(row, col):
                raise ValueError(
                    f"guaranteed-safe cell ({row}, {col}) is outside the board"
                )
        p_mines = self.p_mine_field()
        self._mine_outcomes = (rng.random((self.height, self.width)) < p_mines).astype(
            np.float64
        )
        for row, col in safe_cells:
            self._mine_outcomes[row, col] = 0.0
        for row in range(self.height):
            for col in range(self.width):
                cell = self.cells[row][col]
                cell.is_revealed = False
                cell.has_mine = None
                cell.display_value = 0.0

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def iter_neighbours(self, row: int, col: int):
        """Yield (row, col) for eight-connected neighbours."""
        for dr, dc in _NEIGHBOUR_OFFSETS:
            nr, nc = row + dr, col + dc
            if self.in_bounds(nr, nc):
                yield nr, nc

    @staticmethod
    def neighbour_p_sum(p_mines: np.ndarray, row: int, col: int) -> float:
        """Rounded sum of neighbour p_mine values (classic clue analogue)."""
        height, width = p_mines.shape
        total = 0.0
        for dr, dc in _NEIGHBOUR_OFFSETS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width:
                total += float(p_mines[nr, nc])
        return round(total, 1)

    def cell(self, row: int, col: int) -> Cell:
        return self.cells[row][col]

    def neighbour_mine_count(self, row: int, col: int) -> int:
        """Return the sampled number of neighbouring mines."""
        if self._mine_outcomes is None:
            raise RuntimeError("Call new_episode() before reading hidden outcomes")
        return sum(
            int(self._mine_outcomes[nr, nc])
            for nr, nc in self.iter_neighbours(row, col)
        )

    def clue_value(self, row: int, col: int) -> float:
        """Return the visible clue under the configured rule variant."""
        if self.clue_mode == "prob_sum":
            return self.neighbour_p_sum(self.p_mine_field(), row, col)
        if self.clue_mode == "actual_count":
            return float(self.neighbour_mine_count(row, col))
        raise ValueError(f"Unknown clue_mode: {self.clue_mode!r}")

    def flat_index(self, row: int, col: int) -> int:
        return row * self.width + col

    def from_flat_index(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self.height * self.width:
            raise IndexError(f"action index {index} out of range for board size")
        return divmod(index, self.width)

    def p_mine_field(self) -> np.ndarray:
        return np.array(
            [[self.cells[r][c].p_mine for c in range(self.width)] for r in range(self.height)],
            dtype=np.float32,
        )

    def revealed_mask(self) -> np.ndarray:
        return np.array(
            [[self.cells[r][c].is_revealed for c in range(self.width)] for r in range(self.height)],
            dtype=bool,
        )

    def hidden_mine_mask(self) -> np.ndarray:
        """True where the episode outcome is a mine (only valid after ``new_episode``)."""
        if self._mine_outcomes is None:
            raise RuntimeError("Call new_episode() before reading hidden outcomes")
        return self._mine_outcomes.astype(bool)

    def reveal(self, row: int, col: int) -> RevealResult:
        """Reveal a cell and disclose its hidden mine outcome."""
        if self._mine_outcomes is None:
            raise RuntimeError("Call new_episode() before reveal()")

        cell = self.cell(row, col)
        if cell.is_revealed:
            return RevealResult.NOOP

        cell.is_revealed = True
        has_mine = bool(self._mine_outcomes[row, col])
        cell.has_mine = has_mine

        if has_mine:
            return RevealResult.MINE_HIT

        cell.display_value = self.clue_value(row, col)
        if self.is_win():
            return RevealResult.WIN
        return RevealResult.SAFE

    def is_loss(self) -> bool:
        """True if any revealed cell is a mine."""
        return any(
            cell.is_revealed and cell.has_mine is True
            for row in self.cells
            for cell in row
        )

    def is_win(self) -> bool:
        """True when every non-mine cell is revealed (mines may remain hidden)."""
        if self.is_loss() or self._mine_outcomes is None:
            return False
        safe = ~self.hidden_mine_mask()
        return bool(np.all(self.revealed_mask()[safe]))
