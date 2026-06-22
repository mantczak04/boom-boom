from argparse import Namespace

from experiments.common import build_distribution_kwargs


def test_build_distribution_kwargs_constant_with_p():
    args = Namespace(
        distribution="constant",
        p=0.15,
        low=None,
        high=None,
        sigma=None,
        scale=None,
    )
    assert build_distribution_kwargs(args) == {"p": 0.15}


def test_build_distribution_kwargs_constant_without_p():
    args = Namespace(
        distribution="constant",
        p=None,
        low=None,
        high=None,
        sigma=None,
        scale=None,
    )
    assert build_distribution_kwargs(args) is None


def test_build_distribution_kwargs_uniform_partial():
    args = Namespace(
        distribution="uniform",
        p=None,
        low=0.05,
        high=0.25,
        sigma=None,
        scale=None,
    )
    assert build_distribution_kwargs(args) == {"low": 0.05, "high": 0.25}


def test_build_distribution_kwargs_correlated_partial():
    args = Namespace(
        distribution="correlated",
        p=None,
        low=None,
        high=None,
        sigma=1.5,
        scale=None,
    )
    assert build_distribution_kwargs(args) == {"sigma": 1.5}
