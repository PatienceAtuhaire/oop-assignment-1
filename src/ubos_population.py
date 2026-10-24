"""Population analysis and forecasting tools for the UBOS district mini-project.

The assignment's supplied population series are illustrative and measured in
thousands of people. Additional district data in the notebook are also
illustrative, not official UBOS estimates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean, median, stdev, variance
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DistrictPopulation:
    """A district's annual population series, stored in thousands of people."""

    name: str
    years: IntArray
    populations: FloatArray

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("District name must not be empty.")

        years = np.asarray(self.years, dtype=np.int64).copy()
        populations = np.asarray(self.populations, dtype=np.float64).copy()
        if years.ndim != 1 or populations.ndim != 1:
            raise ValueError("Years and populations must be one-dimensional.")
        if len(years) != len(populations):
            raise ValueError("Years and populations must have equal lengths.")
        if len(years) == 0:
            raise ValueError("A population series must contain at least one value.")
        if not np.all(np.isfinite(populations)):
            raise ValueError("Populations must be finite numbers.")
        if np.any(populations < 0):
            raise ValueError("Populations must not be negative.")
        if np.any(np.diff(years) <= 0):
            raise ValueError("Years must be strictly increasing.")
        years.setflags(write=False)
        populations.setflags(write=False)
        object.__setattr__(self, "years", years)
        object.__setattr__(self, "populations", populations)

    def __repr__(self) -> str:
        return (
            f"DistrictPopulation(name={self.name!r}, "
            f"years={len(self.years)}, "
            f"range={self.years[0]}-{self.years[-1]})"
        )

    def __len__(self) -> int:
        return len(self.populations)


def descriptive_statistics(populations: FloatArray) -> dict[str, float]:
    """Return descriptive statistics from statistics and NumPy.

    ``statistics.variance`` and ``statistics.stdev`` use the sample
    denominator (n - 1). NumPy's defaults use the population denominator n.
    """
    values = np.asarray(populations, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("At least two one-dimensional population values are required.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Populations must be finite numbers.")

    python_values = values.tolist()
    return {
        "statistics_mean": float(mean(python_values)),
        "statistics_median": float(median(python_values)),
        "statistics_variance": float(variance(python_values)),
        "statistics_standard_deviation": float(stdev(python_values)),
        "numpy_mean": float(np.mean(values)),
        "numpy_median": float(np.median(values)),
        "numpy_variance": float(np.var(values)),
        "numpy_standard_deviation": float(np.std(values)),
    }


def year_on_year_growth_rates(populations: FloatArray) -> FloatArray:
    """Return year-on-year percentage growth; undefined zero baselines are rejected."""
    values = np.asarray(populations, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("At least two population values are required.")
    if not np.all(np.isfinite(values)) or np.any(values[:-1] == 0):
        raise ValueError("Growth rates require finite values and nonzero prior populations.")
    return np.diff(values) / values[:-1] * 100


def compound_annual_growth_rate(years: IntArray, populations: FloatArray) -> float:
    """Return CAGR as a percentage, using elapsed calendar years."""
    year_values = np.asarray(years, dtype=np.int64)
    population_values = np.asarray(populations, dtype=np.float64)
    if year_values.ndim != 1 or population_values.ndim != 1:
        raise ValueError("Years and populations must be one-dimensional.")
    if len(year_values) != len(population_values) or len(year_values) < 2:
        raise ValueError("Years and populations must have equal lengths of at least two.")
    elapsed_years = int(year_values[-1] - year_values[0])
    if elapsed_years <= 0 or population_values[0] <= 0 or population_values[-1] <= 0:
        raise ValueError("CAGR requires increasing years and positive first and last populations.")
    if not np.all(np.isfinite(population_values)):
        raise ValueError("Populations must be finite numbers.")
    return float(((population_values[-1] / population_values[0]) ** (1 / elapsed_years) - 1) * 100)


def forecast_metrics(actual: FloatArray, predicted: FloatArray) -> dict[str, float]:
    """Calculate MAE, RMSE and MAPE, with MAPE expressed as a percentage."""
    actual_values = np.asarray(actual, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    if actual_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("Actual and predicted values must be one-dimensional.")
    if len(actual_values) == 0 or actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted values must have the same nonzero length.")
    if not np.all(np.isfinite(actual_values)) or not np.all(np.isfinite(predicted_values)):
        raise ValueError("Actual and predicted values must be finite.")
    if np.any(actual_values == 0):
        raise ValueError("MAPE is undefined for actual values equal to zero.")

    errors = predicted_values - actual_values
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAPE": float(np.mean(np.abs(errors / actual_values)) * 100),
    }


class Forecaster(ABC):
    """Abstract interface shared by the three population forecasting models."""

    model_name: ClassVar[str]

    def __init__(self) -> None:
        self._years: IntArray | None = None
        self._populations: FloatArray | None = None
        self._fitted_values: FloatArray | None = None

    @property
    def fitted_values(self) -> FloatArray:
        """Return model values for the years supplied to :meth:`fit`."""
        if self._fitted_values is None:
            raise RuntimeError("Call fit() before requesting fitted values.")
        return self._fitted_values.copy()

    @abstractmethod
    def fit(self, years: IntArray, populations: FloatArray) -> Forecaster:
        """Fit the model to a chronological population series."""

    @abstractmethod
    def predict(self, horizon: int) -> FloatArray:
        """Predict a positive number of years beyond the fitted series."""

    def _set_training_data(self, years: IntArray, populations: FloatArray) -> None:
        year_values = np.asarray(years, dtype=np.int64)
        population_values = np.asarray(populations, dtype=np.float64)
        if year_values.ndim != 1 or population_values.ndim != 1:
            raise ValueError("Years and populations must be one-dimensional.")
        if len(year_values) != len(population_values) or len(year_values) < 2:
            raise ValueError("At least two matching year and population values are required.")
        if np.any(np.diff(year_values) <= 0):
            raise ValueError("Years must be strictly increasing.")
        if not np.all(np.isfinite(population_values)) or np.any(population_values < 0):
            raise ValueError("Populations must be finite and nonnegative.")
        self._years = year_values.copy()
        self._populations = population_values.copy()

    def _require_fitted(self, horizon: int) -> tuple[IntArray, FloatArray]:
        if self._years is None or self._populations is None:
            raise RuntimeError("Call fit() before predict().")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("Forecast horizon must be a positive integer.")
        return self._years, self._populations


class LinearTrendForecaster(Forecaster):
    """Fit a straight line in calendar year and population using ``np.polyfit``."""

    model_name = "Linear trend"

    def fit(self, years: IntArray, populations: FloatArray) -> LinearTrendForecaster:
        self._set_training_data(years, populations)
        assert self._years is not None and self._populations is not None
        self._slope, self._intercept = np.polyfit(self._years, self._populations, 1)
        self._fitted_values = np.polyval((self._slope, self._intercept), self._years)
        return self

    def predict(self, horizon: int) -> FloatArray:
        years, _ = self._require_fitted(horizon)
        future_years = np.arange(years[-1] + 1, years[-1] + horizon + 1)
        return np.asarray(np.polyval((self._slope, self._intercept), future_years), dtype=np.float64)


class ExponentialGrowthForecaster(Forecaster):
    """Extrapolate a constant annual compound growth rate from the endpoints."""

    model_name = "Exponential/CAGR"

    def fit(self, years: IntArray, populations: FloatArray) -> ExponentialGrowthForecaster:
        self._set_training_data(years, populations)
        assert self._years is not None and self._populations is not None
        if self._populations[0] <= 0 or self._populations[-1] <= 0:
            raise ValueError("Exponential growth requires positive first and last populations.")
        elapsed_years = self._years[-1] - self._years[0]
        self._annual_log_growth = np.log(self._populations[-1] / self._populations[0]) / elapsed_years
        self._fitted_values = self._populations[0] * np.exp(
            self._annual_log_growth * (self._years - self._years[0])
        )
        return self

    def predict(self, horizon: int) -> FloatArray:
        years, populations = self._require_fitted(horizon)
        future_years = np.arange(years[-1] + 1, years[-1] + horizon + 1)
        return np.asarray(
            populations[-1]
            * np.exp(self._annual_log_growth * (future_years - years[-1])),
            dtype=np.float64,
        )


class FibonacciRatioForecaster(Forecaster):
    """Scale each predicted value by successive Fibonacci-number ratios.

    Ratios begin at F3/F2 (2/1), then F4/F3 (3/2), and converge toward the
    golden ratio. This intentionally literal model is a comparison baseline,
    not a defensible demographic assumption.
    """

    model_name = "Fibonacci ratio"

    def fit(self, years: IntArray, populations: FloatArray) -> FibonacciRatioForecaster:
        self._set_training_data(years, populations)
        assert self._years is not None and self._populations is not None
        fitted_values = np.empty_like(self._populations)
        fitted_values[0] = self._populations[0]
        previous, current = 1.0, 2.0
        for index in range(1, len(fitted_values)):
            fitted_values[index] = fitted_values[index - 1] * current / previous
            previous, current = current, previous + current
        self._fitted_values = fitted_values
        return self

    def predict(self, horizon: int) -> FloatArray:
        _, populations = self._require_fitted(horizon)
        previous, current = 1.0, 2.0
        value = float(populations[-1])
        forecasts = np.empty(horizon, dtype=np.float64)
        for index in range(horizon):
            value *= current / previous
            forecasts[index] = value
            previous, current = current, previous + current
        return forecasts


FORECASTER_TYPES: tuple[type[Forecaster], ...] = (
    LinearTrendForecaster,
    ExponentialGrowthForecaster,
    FibonacciRatioForecaster,
)


def bootstrap_prediction_interval(
    forecaster_type: type[Forecaster],
    years: IntArray,
    populations: FloatArray,
    horizon: int,
    *,
    resamples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return point forecasts and residual-bootstrap prediction interval bounds.

    Residuals are sampled independently for each forecast year and added to
    the deterministic forecast. This provides a simple empirical interval;
    it does not model serial correlation or parameter uncertainty.
    """
    if resamples < 1_000:
        raise ValueError("At least 1,000 bootstrap resamples are required.")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between zero and one.")
    model = forecaster_type().fit(years, populations)
    residuals = np.asarray(populations, dtype=np.float64) - model.fitted_values
    if len(residuals) == 0:
        raise ValueError("Residual bootstrap requires fitted residuals.")
    point_forecast = model.predict(horizon)
    rng = np.random.default_rng(seed)
    sampled_residuals = rng.choice(residuals, size=(resamples, horizon), replace=True)
    simulated_forecasts = point_forecast + sampled_residuals
    tail_probability = (1 - confidence) / 2
    lower = np.quantile(simulated_forecasts, tail_probability, axis=0)
    upper = np.quantile(simulated_forecasts, 1 - tail_probability, axis=0)
    return point_forecast, np.asarray(lower), np.asarray(upper)


def additional_classrooms(
    population_2024_thousands: float,
    population_2029_thousands: float,
    *,
    school_age_share: float = 0.18,
    pupils_per_classroom: int = 53,
) -> int:
    """Estimate additional classrooms beyond current 2024 capacity needs.

    Population inputs are in thousands. It is assumed current classrooms meet
    the 2024 school-age demand; each year's required classrooms are rounded up.
    """
    if not np.isfinite(population_2024_thousands) or not np.isfinite(population_2029_thousands):
        raise ValueError("Population estimates must be finite numbers.")
    if not np.isfinite(school_age_share):
        raise ValueError("School-age share must be a finite number.")
    if population_2024_thousands < 0 or population_2029_thousands < 0:
        raise ValueError("Population estimates must be nonnegative.")
    if not 0 <= school_age_share <= 1:
        raise ValueError("School-age share must be between zero and one.")
    if not isinstance(pupils_per_classroom, (int, np.integer)) or pupils_per_classroom < 1:
        raise ValueError("Pupils per classroom must be at least one.")
    current_need = int(np.ceil(population_2024_thousands * 1_000 * school_age_share / pupils_per_classroom))
    future_need = int(np.ceil(population_2029_thousands * 1_000 * school_age_share / pupils_per_classroom))
    return max(0, future_need - current_need)
