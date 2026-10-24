"""Unit tests for the UBOS district population mini-project."""

import numpy as np
import pytest

from src.ubos_population import (
    DistrictPopulation,
    ExponentialGrowthForecaster,
    FibonacciRatioForecaster,
    LinearTrendForecaster,
    additional_classrooms,
    bootstrap_prediction_interval,
    descriptive_statistics,
    forecast_metrics,
)


def test_district_population_validates_and_describes_series() -> None:
    district = DistrictPopulation("Gulu", np.array([2020, 2021]), np.array([300, 310]))

    assert len(district) == 2
    assert "Gulu" in repr(district)
    assert not district.populations.flags.writeable


@pytest.mark.parametrize(
    ("years", "populations"),
    [
        ([2020, 2021], [300]),
        ([2020, 2021], [300, -1]),
    ],
)
def test_district_population_rejects_invalid_series(years, populations) -> None:
    with pytest.raises(ValueError):
        DistrictPopulation("Gulu", np.array(years), np.array(populations))


def test_descriptive_statistics_use_sample_and_population_variance() -> None:
    result = descriptive_statistics(np.array([1.0, 3.0]))

    assert result["statistics_variance"] == 2.0
    assert result["numpy_variance"] == 1.0
    assert result["statistics_mean"] == result["numpy_mean"]


@pytest.mark.parametrize(
    "forecaster_type",
    [LinearTrendForecaster, ExponentialGrowthForecaster, FibonacciRatioForecaster],
)
def test_forecasters_predict_requested_horizon(forecaster_type) -> None:
    model = forecaster_type().fit(np.array([2020, 2021, 2022]), np.array([100.0, 110.0, 120.0]))

    result = model.predict(2)

    assert result.shape == (2,)
    assert np.all(np.isfinite(result))


def test_forecaster_rejects_zero_horizon() -> None:
    model = LinearTrendForecaster().fit(np.array([2020, 2021]), np.array([100.0, 110.0]))

    with pytest.raises(ValueError):
        model.predict(0)


def test_bootstrap_is_reproducible_and_requires_one_thousand_resamples() -> None:
    years = np.arange(2015, 2022)
    populations = np.array([100.0, 104.0, 109.0, 113.0, 119.0, 123.0, 128.0])
    first = bootstrap_prediction_interval(LinearTrendForecaster, years, populations, 2, seed=7)
    second = bootstrap_prediction_interval(LinearTrendForecaster, years, populations, 2, seed=7)

    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    with pytest.raises(ValueError):
        bootstrap_prediction_interval(
            LinearTrendForecaster, years, populations, 2, resamples=999
        )


def test_metrics_and_classroom_edge_cases() -> None:
    metrics = forecast_metrics(np.array([10.0, 20.0]), np.array([12.0, 18.0]))

    assert metrics["MAE"] == 2.0
    assert additional_classrooms(100.0, 100.0) == 0
    with pytest.raises(ValueError):
        forecast_metrics(np.array([0.0]), np.array([1.0]))
