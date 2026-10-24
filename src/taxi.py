"""Taxi-route revenue, demand forecasting, equilibrium, and fleet planning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean, stdev, variance
from typing import Callable, ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Route:
    """Passenger demand observations and fare for a named taxi route."""

    name: str
    passenger_counts: FloatArray
    fare_ugx: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Route name must not be empty.")
        counts = np.asarray(self.passenger_counts, dtype=np.float64).copy()
        if counts.ndim != 1 or len(counts) < 1:
            raise ValueError("Passenger counts must be a nonempty one-dimensional array.")
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("Passenger counts must be finite and nonnegative.")
        if not np.isfinite(self.fare_ugx) or self.fare_ugx < 0:
            raise ValueError("Fare must be finite and nonnegative.")
        counts.setflags(write=False)
        object.__setattr__(self, "passenger_counts", counts)
        object.__setattr__(self, "fare_ugx", float(self.fare_ugx))

    def daily_revenue(self) -> FloatArray:
        """Return observed revenue in UGX for each day."""
        return self.passenger_counts * self.fare_ugx

    def total_revenue(self) -> float:
        """Return total observed revenue in UGX."""
        return float(np.sum(self.daily_revenue()))

    def descriptive_statistics(self) -> dict[str, float]:
        """Return mean, sample variance and sample standard deviation of riders."""
        if len(self.passenger_counts) < 2:
            raise ValueError("At least two observations are required for sample statistics.")
        counts = self.passenger_counts.tolist()
        return {
            "mean": float(mean(counts)),
            "variance": float(variance(counts)),
            "standard_deviation": float(stdev(counts)),
        }


def solve_supply_demand_equilibrium(
    *,
    demand_intercept: float = 120.0,
    demand_slope: float = 0.02,
    supply_intercept: float = 10.0,
    supply_slope: float = 0.03,
    current_fare_ugx: float = 2_000.0,
) -> dict[str, float | str]:
    """Solve linear demand/supply equations for equilibrium Q and fare P.

    Demand is ``Q + demand_slope*P = demand_intercept`` and supply is
    ``Q - supply_slope*P = supply_intercept``.
    """
    values = (
        demand_intercept,
        demand_slope,
        supply_intercept,
        supply_slope,
        current_fare_ugx,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Equilibrium inputs must be finite.")
    if demand_intercept < 0 or supply_intercept < 0 or current_fare_ugx < 0:
        raise ValueError("Demand, supply and current fare values must be nonnegative.")
    if demand_slope <= 0 or supply_slope <= 0:
        raise ValueError("Demand and supply slopes must be positive.")
    matrix = np.array([[1.0, demand_slope], [1.0, -supply_slope]])
    right_hand_side = np.array([demand_intercept, supply_intercept])
    quantity, price = solve(matrix, right_hand_side, assume_a="gen")
    if quantity < 0 or price < 0:
        raise ValueError("The supplied curves do not yield a nonnegative equilibrium.")
    relation = (
        "above" if current_fare_ugx > price else "below" if current_fare_ugx < price else "at"
    )
    return {
        "quantity_passengers_per_trip_hour": float(quantity),
        "equilibrium_fare_ugx": float(price),
        "current_fare_ugx": float(current_fare_ugx),
        "current_fare_relation": relation,
    }


class Forecaster(ABC):
    """Base interface for route passenger forecasting models."""

    model_name: ClassVar[str]

    @abstractmethod
    def fit(self, passenger_counts: FloatArray) -> Forecaster:
        """Fit a model on chronologically ordered observed passenger counts."""

    @abstractmethod
    def predict(self, horizon: int = 1) -> FloatArray:
        """Predict one or more future passenger counts."""


class MovingAverageForecaster(Forecaster):
    """Forecast with the mean of the most recent three observations."""

    model_name = "3-day moving average"

    def __init__(self, window: int = 3) -> None:
        if not isinstance(window, (int, np.integer)) or window < 1:
            raise ValueError("Moving-average window must be a positive integer.")
        self.window = int(window)
        self._observations: FloatArray | None = None

    def fit(self, passenger_counts: FloatArray) -> MovingAverageForecaster:
        values = _validate_passenger_counts(passenger_counts)
        if len(values) < self.window:
            raise ValueError("At least one full moving-average window is required.")
        self._observations = values.copy()
        return self

    def predict(self, horizon: int = 1) -> FloatArray:
        values = _require_fitted(self._observations, horizon)
        forecast = float(np.mean(values[-self.window :]))
        return np.full(horizon, forecast, dtype=np.float64)


class ExponentialSmoothingForecaster(Forecaster):
    """Simple exponential smoothing with a tunable smoothing parameter alpha."""

    model_name = "Simple exponential smoothing"

    def __init__(self, alpha: float = 0.3) -> None:
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("Alpha must be finite and in (0, 1].")
        self.alpha = float(alpha)
        self._observations: FloatArray | None = None
        self._level: float | None = None

    def fit(self, passenger_counts: FloatArray) -> ExponentialSmoothingForecaster:
        values = _validate_passenger_counts(passenger_counts)
        if len(values) < 1:
            raise ValueError("At least one observation is required.")
        level = float(values[0])
        for observation in values[1:]:
            level = self.alpha * float(observation) + (1 - self.alpha) * level
        self._observations = values.copy()
        self._level = level
        return self

    def predict(self, horizon: int = 1) -> FloatArray:
        _require_fitted(self._observations, horizon)
        if self._level is None:
            raise RuntimeError("Call fit() before predict().")
        return np.full(horizon, self._level, dtype=np.float64)


class LinearTrendForecaster(Forecaster):
    """Fit a least-squares linear trend to daily passenger counts."""

    model_name = "Linear trend"

    def __init__(self) -> None:
        self._observations: FloatArray | None = None
        self._slope: float | None = None
        self._intercept: float | None = None

    def fit(self, passenger_counts: FloatArray) -> LinearTrendForecaster:
        values = _validate_passenger_counts(passenger_counts)
        if len(values) < 2:
            raise ValueError("Linear trend requires at least two observations.")
        self._observations = values.copy()
        self._slope, self._intercept = np.polyfit(np.arange(len(values)), values, 1)
        return self

    def predict(self, horizon: int = 1) -> FloatArray:
        values = _require_fitted(self._observations, horizon)
        if self._slope is None or self._intercept is None:
            raise RuntimeError("Call fit() before predict().")
        future_days = np.arange(len(values), len(values) + horizon)
        return np.maximum(
            np.asarray(self._slope * future_days + self._intercept, dtype=np.float64),
            0.0,
        )


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the observations from the most recent complete seasonal cycle."""

    model_name = "7-day seasonal naive"

    def __init__(self, season_length: int = 7) -> None:
        if not isinstance(season_length, (int, np.integer)) or season_length < 1:
            raise ValueError("Season length must be a positive integer.")
        self.season_length = int(season_length)
        self._observations: FloatArray | None = None

    def fit(self, passenger_counts: FloatArray) -> SeasonalNaiveForecaster:
        values = _validate_passenger_counts(passenger_counts)
        if len(values) < self.season_length:
            raise ValueError("At least one complete season is required.")
        self._observations = values.copy()
        return self

    def predict(self, horizon: int = 1) -> FloatArray:
        values = _require_fitted(self._observations, horizon)
        seasonal_values = values[-self.season_length :]
        return np.resize(seasonal_values, horizon).astype(np.float64)


def _validate_passenger_counts(passenger_counts: FloatArray) -> FloatArray:
    values = np.asarray(passenger_counts, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Passenger counts must be a nonempty one-dimensional series.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Passenger counts must be finite and nonnegative.")
    return values


def _require_fitted(observations: FloatArray | None, horizon: int) -> FloatArray:
    if observations is None:
        raise RuntimeError("Call fit() before predict().")
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("Forecast horizon must be a positive integer.")
    return observations


def tune_exponential_smoothing_alpha(
    passenger_counts: FloatArray,
    *,
    alphas: FloatArray | None = None,
    initial_training_size: int = 3,
) -> tuple[float, dict[float, float]]:
    """Choose alpha by rolling-origin MAE within the supplied series.

    Each validation prediction uses only observations preceding its target
    day. The grid defaults to alpha values from 0.1 to 1.0 in increments of
    0.1. Ties are resolved by choosing the smaller alpha.
    """
    values = _validate_passenger_counts(passenger_counts)
    alpha_values = np.asarray(
        np.arange(0.1, 1.01, 0.1) if alphas is None else alphas,
        dtype=np.float64,
    )
    if (
        alpha_values.ndim != 1
        or len(alpha_values) == 0
        or not np.all(np.isfinite(alpha_values))
        or np.any(alpha_values <= 0)
        or np.any(alpha_values > 1)
    ):
        raise ValueError("Alpha candidates must be a nonempty array in (0, 1].")
    if (
        not isinstance(initial_training_size, (int, np.integer))
        or initial_training_size < 1
        or initial_training_size >= len(values)
    ):
        raise ValueError("Initial training size must be at least one and leave validation data.")
    errors_by_alpha = {}
    for alpha in alpha_values:
        absolute_errors = []
        for target_index in range(initial_training_size, len(values)):
            model = ExponentialSmoothingForecaster(float(alpha)).fit(
                values[:target_index]
            )
            prediction = model.predict()[0]
            absolute_errors.append(abs(prediction - values[target_index]))
        errors_by_alpha[float(alpha)] = float(np.mean(absolute_errors))
    best_alpha = min(errors_by_alpha, key=lambda alpha: (errors_by_alpha[alpha], alpha))
    return best_alpha, errors_by_alpha


def rolling_origin_mae(
    forecaster_factory: Callable[[], Forecaster],
    passenger_counts: FloatArray,
    *,
    first_test_day: int = 4,
) -> float:
    """Return walk-forward one-day MAE starting at one-based ``first_test_day``."""
    values = _validate_passenger_counts(passenger_counts)
    if (
        not isinstance(first_test_day, (int, np.integer))
        or first_test_day < 2
        or first_test_day > len(values)
    ):
        raise ValueError("First test day must be between day 2 and the final observed day.")
    errors = []
    for target_index in range(first_test_day - 1, len(values)):
        model = forecaster_factory()
        model.fit(values[:target_index])
        errors.append(abs(float(model.predict()[0]) - values[target_index]))
    return float(np.mean(errors))


def plan_fleet(
    forecast_passengers: float,
    *,
    trips_per_vehicle_per_day: int = 8,
    seats_per_vehicle: int = 14,
    buffer_fraction: float = 0.15,
) -> dict[str, int | float]:
    """Compute buffered day demand, trips and whole-vehicle fleet requirement.

    Forecast passengers are interpreted as one-way passenger trips per day
    across the route. The 15% buffer is added to forecast demand, and vehicles
    are rounded up so that capacity is never below buffered demand.
    """
    if not np.isfinite(forecast_passengers) or forecast_passengers < 0:
        raise ValueError("Forecast passengers must be finite and nonnegative.")
    if (
        not isinstance(trips_per_vehicle_per_day, (int, np.integer))
        or trips_per_vehicle_per_day < 1
        or not isinstance(seats_per_vehicle, (int, np.integer))
        or seats_per_vehicle < 1
    ):
        raise ValueError("Trips and seats must be positive integers.")
    if not np.isfinite(buffer_fraction) or buffer_fraction < 0:
        raise ValueError("Buffer fraction must be finite and nonnegative.")
    buffered_demand = forecast_passengers * (1 + buffer_fraction)
    one_vehicle_capacity = trips_per_vehicle_per_day * seats_per_vehicle
    vehicles = int(np.ceil(buffered_demand / one_vehicle_capacity))
    return {
        "forecast_passengers": float(forecast_passengers),
        "buffered_passengers": float(buffered_demand),
        "one_vehicle_daily_capacity": one_vehicle_capacity,
        "vehicles": vehicles,
        "total_seat_trip_capacity": vehicles * one_vehicle_capacity,
    }


def generate_weekly_passenger_counts(
    days: int = 60,
    *,
    seed: int = 2026,
    base_demand: float = 50.0,
    noise_standard_deviation: float = 3.0,
) -> FloatArray:
    """Generate seeded synthetic weekly demand with Friday peaks and Sunday dips."""
    if not isinstance(days, (int, np.integer)) or days < 1:
        raise ValueError("Days must be a positive integer.")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("Seed must be an integer.")
    if (
        not np.isfinite(base_demand)
        or base_demand < 0
        or not np.isfinite(noise_standard_deviation)
        or noise_standard_deviation < 0
    ):
        raise ValueError("Demand and noise parameters must be finite and nonnegative.")
    weekday_effects = np.array([0.0, -2.0, 1.0, 2.0, 15.0, 7.0, -14.0])
    rng = np.random.default_rng(seed)
    daily_noise = rng.normal(0, noise_standard_deviation, size=days)
    return np.maximum(
        0,
        base_demand + weekday_effects[np.arange(days) % 7] + daily_noise,
    )
