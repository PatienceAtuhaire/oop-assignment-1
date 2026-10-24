"""Unit tests for the fish-stock and export-revenue risk mini-project."""

import numpy as np
import pytest

from src.fishery import (
    FishStock,
    PriceModel,
    RiskAssessor,
    value_at_risk,
    weekly_revenue_ugx,
)


def test_stock_simulation_preserves_initial_value_and_harvest_accounting() -> None:
    simulation = FishStock().simulate(0.2, weeks=52)

    assert simulation.stock_tonnes[0] == 4_000
    assert len(simulation.stock_tonnes) == 53
    assert len(simulation.harvest_tonnes) == 52
    assert simulation.final_stock_tonnes > 0
    assert simulation.total_harvest_tonnes > 0
    assert FishStock().maximum_sustainable_yield_tonnes_per_week == 1_000


@pytest.mark.parametrize("harvest_rate", [-0.1, 1.1])
def test_stock_rejects_invalid_harvest_rates(harvest_rate) -> None:
    with pytest.raises(ValueError):
        FishStock().simulate(harvest_rate)


def test_closed_season_suspends_harvest_for_eight_weeks_per_year() -> None:
    simulation = FishStock().simulate(0.2, weeks=104, closed_season_weeks=8)

    assert np.all(simulation.harvest_tonnes[:8] == 0)
    assert np.all(simulation.harvest_tonnes[52:60] == 0)
    assert np.all(simulation.harvest_tonnes[8:52] > 0)


def test_price_paths_are_seeded_and_bounded() -> None:
    model = PriceModel()
    first = model.simulate_paths(1_000, 52, seed=14)
    second = model.simulate_paths(1_000, 52, seed=14)

    assert np.array_equal(first, second)
    assert np.all(first[:, 0] == 12_000)
    assert np.all((first >= 9_000) & (first <= 16_000))


def test_revenue_converts_tonnes_to_kilograms_and_supports_price_paths() -> None:
    revenue = weekly_revenue_ugx(
        np.array([2.0, 3.0]),
        np.array([[12_000.0, 10_000.0], [11_000.0, 9_000.0]]),
    )

    assert revenue.shape == (2, 2)
    assert revenue[0, 0] == 24_000_000
    assert revenue[1, 1] == 27_000_000


def test_risk_statistics_use_variance_units_and_cv_categories() -> None:
    assessor = RiskAssessor()
    statistics = assessor.revenue_statistics(np.array([90.0, 110.0]))

    assert statistics["mean_ugx"] == 100
    assert statistics["variance_ugx_squared"] == 200
    assert statistics["coefficient_of_variation"] == pytest.approx(0.1414213562)
    assert assessor.classify(statistics["coefficient_of_variation"]) == "Moderate"


def test_risk_assessor_rejects_empty_singleton_and_zero_mean_revenues() -> None:
    assessor = RiskAssessor()
    for values in (np.array([]), np.array([2.0]), np.array([0.0, 0.0])):
        with pytest.raises(ValueError):
            assessor.revenue_statistics(values)


def test_value_at_risk_is_lower_five_percent_revenue_quantile() -> None:
    revenues = np.arange(1_000, dtype=float)

    assert value_at_risk(revenues) == pytest.approx(49.95)
    with pytest.raises(ValueError):
        value_at_risk(revenues, confidence=1.0)
