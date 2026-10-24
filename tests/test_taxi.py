"""Unit tests for taxi route forecasting and fleet planning."""

import numpy as np
import pytest

from src.taxi import (
    ExponentialSmoothingForecaster,
    LinearTrendForecaster,
    MovingAverageForecaster,
    Route,
    SeasonalNaiveForecaster,
    generate_weekly_passenger_counts,
    plan_fleet,
    rolling_origin_mae,
    solve_supply_demand_equilibrium,
    tune_exponential_smoothing_alpha,
)


def test_route_revenue_and_descriptive_statistics() -> None:
    route = Route("Ntinda", np.array([10, 20, 30]), fare_ugx=2_000)

    assert route.daily_revenue().tolist() == [20_000, 40_000, 60_000]
    assert route.total_revenue() == 120_000
    assert route.descriptive_statistics()["mean"] == 20
    assert route.descriptive_statistics()["variance"] == 100


@pytest.mark.parametrize(
    ("counts", "fare"),
    [
        (np.array([]), 2_000),
        (np.array([1.0, -1.0]), 2_000),
        (np.array([1.0]), -1),
    ],
)
def test_route_rejects_invalid_values(counts, fare) -> None:
    with pytest.raises(ValueError):
        Route("Invalid", counts, fare)


def test_supply_demand_equilibrium_uses_scipy_system_and_fare_relation() -> None:
    equilibrium = solve_supply_demand_equilibrium(current_fare_ugx=2_000)

    assert equilibrium["quantity_passengers_per_trip_hour"] == pytest.approx(76)
    assert equilibrium["equilibrium_fare_ugx"] == pytest.approx(2_200)
    assert equilibrium["current_fare_relation"] == "below"


@pytest.mark.parametrize(
    "forecaster",
    [
        MovingAverageForecaster(),
        ExponentialSmoothingForecaster(alpha=0.5),
        LinearTrendForecaster(),
        SeasonalNaiveForecaster(),
    ],
)
def test_forecasters_fit_and_return_horizon(forecaster) -> None:
    values = np.array([10.0, 12.0, 14.0, 13.0, 15.0, 20.0, 18.0])

    predictions = forecaster.fit(values).predict(2)

    assert predictions.shape == (2,)
    assert np.all(np.isfinite(predictions))


def test_forecasters_reject_invalid_horizon_and_bad_alpha() -> None:
    with pytest.raises(ValueError):
        ExponentialSmoothingForecaster(alpha=0)
    with pytest.raises(ValueError):
        MovingAverageForecaster().fit(np.array([1.0, 2.0, 3.0])).predict(0)


def test_rolling_origin_and_alpha_grid_are_reproducible() -> None:
    values = np.array([35, 40, 42, 50, 55, 60, 48, 52, 47, 45], dtype=float)
    best_alpha, errors = tune_exponential_smoothing_alpha(values[:4])

    assert best_alpha in errors
    assert len(errors) == 10
    mae = rolling_origin_mae(
        lambda: ExponentialSmoothingForecaster(alpha=best_alpha),
        values,
        first_test_day=4,
    )
    assert mae >= 0


def test_fleet_rounds_up_and_handles_zero_demand() -> None:
    plan = plan_fleet(100)

    assert plan["buffered_passengers"] == pytest.approx(115)
    assert plan["one_vehicle_daily_capacity"] == 112
    assert plan["vehicles"] == 2
    assert plan_fleet(0)["vehicles"] == 0


def test_weekly_synthetic_counts_are_seeded_and_show_friday_sunday_effects() -> None:
    demand = generate_weekly_passenger_counts(700, seed=17, noise_standard_deviation=0)

    assert np.array_equal(demand, generate_weekly_passenger_counts(700, seed=17, noise_standard_deviation=0))
    weekday_means = [float(np.mean(demand[day::7])) for day in range(7)]
    assert weekday_means[4] > max(weekday_means[:4])
    assert weekday_means[6] < min(weekday_means[:6])


def test_seasonal_naive_recovers_next_weekday_value() -> None:
    values = np.array([10, 11, 12, 13, 20, 15, 5, 10, 11, 12, 13, 20, 15, 5], dtype=float)

    assert SeasonalNaiveForecaster().fit(values).predict()[0] == 10
