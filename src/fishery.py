"""Fish-stock dynamics, export-price simulation, and revenue-risk analysis."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, stdev, variance

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FishStockSimulation:
    """Weekly stock trajectory and harvested biomass for one scenario."""

    stock_tonnes: FloatArray
    harvest_tonnes: FloatArray
    harvest_rate: float
    closed_season_weeks: int

    def __post_init__(self) -> None:
        stocks = np.asarray(self.stock_tonnes, dtype=np.float64).copy()
        harvests = np.asarray(self.harvest_tonnes, dtype=np.float64).copy()
        if stocks.ndim != 1 or harvests.ndim != 1:
            raise ValueError("Stock and harvest trajectories must be one-dimensional.")
        if len(stocks) != len(harvests) + 1:
            raise ValueError("Stock trajectory must have one more entry than harvest.")
        if not np.all(np.isfinite(stocks)) or not np.all(np.isfinite(harvests)):
            raise ValueError("Stock and harvest trajectories must be finite.")
        if np.any(stocks < 0) or np.any(harvests < 0):
            raise ValueError("Stock and harvest trajectories must be nonnegative.")
        stocks.setflags(write=False)
        harvests.setflags(write=False)
        object.__setattr__(self, "stock_tonnes", stocks)
        object.__setattr__(self, "harvest_tonnes", harvests)

    @property
    def final_stock_tonnes(self) -> float:
        """Stock remaining after the simulated period."""
        return float(self.stock_tonnes[-1])

    @property
    def total_harvest_tonnes(self) -> float:
        """Cumulative harvested biomass over the simulated period."""
        return float(np.sum(self.harvest_tonnes))


class FishStock:
    """Simulate discrete logistic fish-stock growth with proportional harvesting.

    The state update follows the assignment formula::

        N(t+1) = N(t) + r*N(t)*(1 - N(t)/K) - h*N(t)

    The harvest is taken from the current week's stock, then natural growth
    and density dependence are applied. Values are measured in tonnes and
    the simulation time step is one week.
    """

    def __init__(
        self,
        initial_stock_tonnes: float = 4_000.0,
        growth_rate: float = 0.4,
        carrying_capacity_tonnes: float = 10_000.0,
    ) -> None:
        values = (initial_stock_tonnes, growth_rate, carrying_capacity_tonnes)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Stock parameters must be finite.")
        if initial_stock_tonnes < 0:
            raise ValueError("Initial stock must be nonnegative.")
        if growth_rate <= 0:
            raise ValueError("Growth rate must be positive.")
        if carrying_capacity_tonnes <= 0:
            raise ValueError("Carrying capacity must be positive.")
        if initial_stock_tonnes > carrying_capacity_tonnes:
            raise ValueError("Initial stock must not exceed carrying capacity.")
        self.initial_stock_tonnes = float(initial_stock_tonnes)
        self.growth_rate = float(growth_rate)
        self.carrying_capacity_tonnes = float(carrying_capacity_tonnes)

    @property
    def maximum_sustainable_yield_tonnes_per_week(self) -> float:
        """The continuous-logistic benchmark MSY, rK/4, per model time step."""
        return self.growth_rate * self.carrying_capacity_tonnes / 4

    def simulate(
        self,
        harvest_rate: float,
        weeks: int = 52,
        *,
        closed_season_weeks: int = 0,
    ) -> FishStockSimulation:
        """Simulate weekly biomass, optionally closing harvest early each year."""
        if not np.isfinite(harvest_rate) or not 0 <= harvest_rate <= 1:
            raise ValueError("Harvest rate must be finite and between zero and one.")
        if not isinstance(weeks, (int, np.integer)) or weeks < 1:
            raise ValueError("Weeks must be a positive integer.")
        if (
            not isinstance(closed_season_weeks, (int, np.integer))
            or not 0 <= closed_season_weeks <= 52
        ):
            raise ValueError("Closed-season duration must be an integer from 0 to 52.")

        stocks = np.empty(weeks + 1, dtype=np.float64)
        harvests = np.empty(weeks, dtype=np.float64)
        stocks[0] = self.initial_stock_tonnes
        for week in range(weeks):
            is_closed = week % 52 < closed_season_weeks
            weekly_rate = 0.0 if is_closed else harvest_rate
            current_stock = stocks[week]
            harvests[week] = weekly_rate * current_stock
            next_stock = (
                current_stock
                + self.growth_rate
                * current_stock
                * (1 - current_stock / self.carrying_capacity_tonnes)
                - harvests[week]
            )
            stocks[week + 1] = np.clip(next_stock, 0, self.carrying_capacity_tonnes)
        return FishStockSimulation(
            stocks,
            harvests,
            float(harvest_rate),
            int(closed_season_weeks),
        )


class PriceModel:
    """Generate bounded seeded weekly export prices using an additive random walk."""

    def __init__(
        self,
        starting_price_ugx_per_kg: float = 12_000.0,
        lower_bound_ugx_per_kg: float = 9_000.0,
        upper_bound_ugx_per_kg: float = 16_000.0,
        weekly_volatility_ugx_per_kg: float = 350.0,
    ) -> None:
        values = (
            starting_price_ugx_per_kg,
            lower_bound_ugx_per_kg,
            upper_bound_ugx_per_kg,
            weekly_volatility_ugx_per_kg,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Price-model parameters must be finite.")
        if lower_bound_ugx_per_kg <= 0:
            raise ValueError("The lower price bound must be positive.")
        if upper_bound_ugx_per_kg <= lower_bound_ugx_per_kg:
            raise ValueError("The upper price bound must exceed the lower bound.")
        if not lower_bound_ugx_per_kg <= starting_price_ugx_per_kg <= upper_bound_ugx_per_kg:
            raise ValueError("Starting price must lie within the price bounds.")
        if weekly_volatility_ugx_per_kg < 0:
            raise ValueError("Weekly volatility must be nonnegative.")
        self.starting_price_ugx_per_kg = float(starting_price_ugx_per_kg)
        self.lower_bound_ugx_per_kg = float(lower_bound_ugx_per_kg)
        self.upper_bound_ugx_per_kg = float(upper_bound_ugx_per_kg)
        self.weekly_volatility_ugx_per_kg = float(weekly_volatility_ugx_per_kg)

    def simulate_paths(
        self,
        paths: int = 1_000,
        weeks: int = 52,
        *,
        seed: int = 2026,
    ) -> FloatArray:
        """Return a ``(paths, weeks)`` bounded random walk, including week 1 start."""
        if not isinstance(paths, (int, np.integer)) or paths < 1:
            raise ValueError("Paths must be a positive integer.")
        if not isinstance(weeks, (int, np.integer)) or weeks < 1:
            raise ValueError("Weeks must be a positive integer.")
        if not isinstance(seed, (int, np.integer)):
            raise ValueError("Seed must be an integer.")
        prices = np.empty((paths, weeks), dtype=np.float64)
        prices[:, 0] = self.starting_price_ugx_per_kg
        rng = np.random.default_rng(seed)
        for week in range(1, weeks):
            innovations = rng.normal(
                0, self.weekly_volatility_ugx_per_kg, size=paths
            )
            prices[:, week] = np.clip(
                prices[:, week - 1] + innovations,
                self.lower_bound_ugx_per_kg,
                self.upper_bound_ugx_per_kg,
            )
        return prices

    def simulate(
        self,
        weeks: int = 52,
        *,
        seed: int = 2026,
    ) -> FloatArray:
        """Return one seeded weekly price path."""
        return self.simulate_paths(paths=1, weeks=weeks, seed=seed)[0]


class RiskAssessor:
    """Summarize revenue and classify its relative variability using CV.

    The default boundaries (CV below 10%, 10–20%, and 20% or more) are
    transparent planning heuristics for this exercise, not regulatory or
    universal fishery-risk thresholds.
    """

    def __init__(self, low_cv_threshold: float = 0.10, high_cv_threshold: float = 0.20):
        if not np.isfinite(low_cv_threshold) or not np.isfinite(high_cv_threshold):
            raise ValueError("CV thresholds must be finite.")
        if low_cv_threshold < 0 or high_cv_threshold <= low_cv_threshold:
            raise ValueError("CV thresholds must be ordered and nonnegative.")
        self.low_cv_threshold = float(low_cv_threshold)
        self.high_cv_threshold = float(high_cv_threshold)

    def revenue_statistics(self, revenue_ugx: FloatArray) -> dict[str, float]:
        """Return statistics-module summaries for at least two revenue values."""
        values = np.asarray(revenue_ugx, dtype=np.float64)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError("At least two one-dimensional revenue values are required.")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Revenue values must be finite and nonnegative.")
        average = float(mean(values.tolist()))
        standard_deviation = float(stdev(values.tolist()))
        if average == 0:
            raise ValueError("Revenue coefficient of variation is undefined when mean is zero.")
        return {
            "mean_ugx": average,
            "median_ugx": float(median(values.tolist())),
            "variance_ugx_squared": float(variance(values.tolist())),
            "standard_deviation_ugx": standard_deviation,
            "coefficient_of_variation": standard_deviation / average,
        }

    def classify(self, coefficient_of_variation: float) -> str:
        """Classify revenue variability as low, moderate or high."""
        if not np.isfinite(coefficient_of_variation) or coefficient_of_variation < 0:
            raise ValueError("Coefficient of variation must be finite and nonnegative.")
        if coefficient_of_variation < self.low_cv_threshold:
            return "Low"
        if coefficient_of_variation < self.high_cv_threshold:
            return "Moderate"
        return "High"


def weekly_revenue_ugx(
    harvest_tonnes: FloatArray,
    prices_ugx_per_kg: FloatArray,
) -> FloatArray:
    """Convert harvested tonnes to kg and calculate weekly revenue in UGX."""
    harvests = np.asarray(harvest_tonnes, dtype=np.float64)
    prices = np.asarray(prices_ugx_per_kg, dtype=np.float64)
    if harvests.ndim != 1 or not np.all(np.isfinite(harvests)):
        raise ValueError("Harvest must be a finite one-dimensional array.")
    if np.any(harvests < 0):
        raise ValueError("Harvest values must be nonnegative.")
    if prices.ndim == 1:
        if len(prices) != len(harvests):
            raise ValueError("Price and harvest arrays must have matching weeks.")
    elif prices.ndim == 2:
        if prices.shape[1] != len(harvests):
            raise ValueError("Each price path must match the harvest duration.")
    else:
        raise ValueError("Prices must be a one- or two-dimensional array.")
    if not np.all(np.isfinite(prices)) or np.any(prices < 0):
        raise ValueError("Prices must be finite and nonnegative.")
    return harvests * 1_000 * prices


def value_at_risk(revenue_ugx: FloatArray, confidence: float = 0.95) -> float:
    """Return lower-tail revenue quantile at ``1 - confidence`` (e.g. 5% VaR)."""
    values = np.asarray(revenue_ugx, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Revenue must be a nonempty finite one-dimensional array.")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between zero and one.")
    return float(np.quantile(values, 1 - confidence))
