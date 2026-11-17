"""Shared fixtures for the test suite."""

import pytest

from alphalab.data import MarketConfig, generate_market


@pytest.fixture(scope="session")
def small_panel():
    """A small synthetic panel reused across all tests."""
    return generate_market(MarketConfig(n_instruments=30, n_days=300, seed=99))


@pytest.fixture(scope="session")
def medium_panel():
    """A medium synthetic panel for integration tests."""
    return generate_market(MarketConfig(n_instruments=80, n_days=600, seed=42))
