"""Tests for mine probability distributions."""

import numpy as np
import pytest

from prob_minesweeper.distributions import (
    ConstantDistribution,
    CorrelatedDistribution,
    UniformDistribution,
    make_distribution,
)


def _assert_valid_field(field: np.ndarray, height: int, width: int) -> None:
    assert field.shape == (height, width)
    assert field.dtype == np.float32
    assert np.all(field >= 0.0)
    assert np.all(field <= 1.0)


@pytest.mark.parametrize(
    ("dist", "kwargs"),
    [
        (CorrelatedDistribution, {}),
        (CorrelatedDistribution, {"sigma": 1.0, "scale": 0.5}),
        (UniformDistribution, {"low": 0.1, "high": 0.4}),
        (ConstantDistribution, {"p": 0.35}),
    ],
)
def test_generate_shape_and_range(dist, kwargs) -> None:
    rng = np.random.default_rng(0)
    field = dist(**kwargs).generate(5, 7, rng)
    _assert_valid_field(field, 5, 7)


def test_constant_distribution_value() -> None:
    field = ConstantDistribution(p=0.42).generate(3, 4, np.random.default_rng(0))
    np.testing.assert_allclose(field, 0.42)


def test_uniform_distribution_bounds() -> None:
    rng = np.random.default_rng(1)
    field = UniformDistribution(low=0.2, high=0.3).generate(20, 20, rng)
    assert field.min() >= 0.2
    assert field.max() <= 0.3


def test_uniform_invalid_range() -> None:
    with pytest.raises(ValueError, match="low"):
        UniformDistribution(low=0.8, high=0.2)


def test_correlated_reproducible_with_seed() -> None:
    a = CorrelatedDistribution().generate(8, 8, np.random.default_rng(42))
    b = CorrelatedDistribution().generate(8, 8, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_correlated_spatial_structure() -> None:
    """Blurred field should correlate more with neighbours than shuffled noise."""
    rng = np.random.default_rng(0)
    field = CorrelatedDistribution(sigma=2.0).generate(32, 32, rng)
    neighbour_delta = np.abs(field[:-1, :] - field[1:, :]).mean()
    shuffled = np.random.default_rng(1).permutation(field.ravel()).reshape(field.shape)
    shuffled_delta = np.abs(shuffled[:-1, :] - shuffled[1:, :]).mean()
    assert neighbour_delta < shuffled_delta


def test_make_distribution_by_name() -> None:
    dist = make_distribution("correlated", sigma=1.5)
    assert isinstance(dist, CorrelatedDistribution)
    assert dist.sigma == 1.5


def test_make_distribution_pass_through() -> None:
    original = ConstantDistribution(p=0.1)
    assert make_distribution(original) is original


def test_make_distribution_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown distribution"):
        make_distribution("gaussian")
