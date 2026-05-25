"""Mine probability distributions for board generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

_DISTRIBUTION_NAMES = ("correlated", "uniform", "constant")


def _clip_probabilities(field: np.ndarray) -> np.ndarray:
    return np.clip(field, 0.0, 1.0).astype(np.float32)


class MineDistribution(ABC):
    """Generates a per-cell mine probability field for one episode."""

    @abstractmethod
    def generate(self, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
        """Return float32 array of shape (H, W) with values in [0, 1]."""


@dataclass(frozen=True)
class CorrelatedDistribution(MineDistribution):
    """Spatially correlated field via Gaussian-blurred white noise."""

    sigma: float = 2.0
    scale: float = 1.0

    def generate(self, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
        noise = rng.standard_normal((height, width)).astype(np.float64)
        blurred = gaussian_filter(noise, sigma=self.sigma, mode="reflect")
        lo, hi = float(blurred.min()), float(blurred.max())
        if hi > lo:
            field = (blurred - lo) / (hi - lo)
        else:
            field = np.full((height, width), 0.5, dtype=np.float64)
        if self.scale != 1.0:
            field = 0.5 + self.scale * (field - 0.5)
        return _clip_probabilities(field)


@dataclass(frozen=True)
class UniformDistribution(MineDistribution):
    """Independent uniform mine probabilities per cell."""

    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high})")

    def generate(self, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
        low = float(np.clip(self.low, 0.0, 1.0))
        high = float(np.clip(self.high, 0.0, 1.0))
        field = rng.uniform(low, high, size=(height, width))
        return _clip_probabilities(field)


@dataclass(frozen=True)
class ConstantDistribution(MineDistribution):
    """Same mine probability on every cell."""

    p: float = 0.2

    def generate(self, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
        del rng
        p = float(np.clip(self.p, 0.0, 1.0))
        return np.full((height, width), p, dtype=np.float32)


def make_distribution(
    distribution: str | MineDistribution,
    **kwargs: Any,
) -> MineDistribution:
    """Build a distribution from a name or pass through an existing instance."""
    if isinstance(distribution, MineDistribution):
        return distribution
    name = distribution.lower()
    if name == "correlated":
        return CorrelatedDistribution(**kwargs)
    if name == "uniform":
        return UniformDistribution(**kwargs)
    if name == "constant":
        return ConstantDistribution(**kwargs)
    valid = ", ".join(_DISTRIBUTION_NAMES)
    raise ValueError(f"Unknown distribution {distribution!r}; expected one of: {valid}")


__all__ = [
    "MineDistribution",
    "CorrelatedDistribution",
    "UniformDistribution",
    "ConstantDistribution",
    "make_distribution",
]
