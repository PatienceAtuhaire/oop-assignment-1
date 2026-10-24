"""Micro-grid demand solving, dispatch statistics, and cost calculations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev, variance
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import solve
from scipy.optimize import nnls


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DaySolution:
    """A daily source allocation and whether its exact solution is feasible."""

    usage: FloatArray
    feasible: bool
    method: str
    residual_norm: float

    def __post_init__(self) -> None:
        usage = np.asarray(self.usage, dtype=np.float64).copy()
        if usage.ndim != 1 or len(usage) < 2:
            raise ValueError("Usage must contain at least two source values.")
        if not np.all(np.isfinite(usage)):
            raise ValueError("Usage values must be finite.")
        usage.setflags(write=False)
        object.__setattr__(self, "usage", usage)

    @property
    def solar(self) -> float:
        """Solar allocation in kWh."""
        return float(self.usage[0])

    @property
    def battery(self) -> float:
        """Battery allocation in kWh."""
        return float(self.usage[1])

    @property
    def diesel(self) -> float:
        """Diesel allocation in kWh for the hybrid-grid solution."""
        if len(self.usage) < 3:
            raise AttributeError("This solution does not include a diesel source.")
        return float(self.usage[2])


class MicroGrid:
    """Solve the two-equation solar and battery demand system.

    The default equations are ``3x + 2y = d1`` and ``4x + y = d2``.
    Negative exact allocations are flagged and handled with non-negative
    least squares (NNLS), which reports its residual rather than pretending
    the original equalities were still satisfied.
    """

    DEFAULT_MATRIX: FloatArray = np.array([[3.0, 2.0], [4.0, 1.0]])
    SOURCE_NAMES = ("solar", "battery")

    def __init__(self, coefficient_matrix: FloatArray | None = None) -> None:
        matrix = np.asarray(
            self.DEFAULT_MATRIX if coefficient_matrix is None else coefficient_matrix,
            dtype=np.float64,
        ).copy()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("The coefficient matrix must be square.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("The coefficient matrix must contain finite values.")
        if matrix.shape[0] != len(self.SOURCE_NAMES):
            raise ValueError("The coefficient matrix has the wrong number of sources.")
        matrix.setflags(write=False)
        self._matrix = matrix

    @property
    def coefficient_matrix(self) -> FloatArray:
        """Return a read-only copy of the coefficient matrix."""
        return self._matrix.copy()

    @property
    def well_posedness(self) -> dict[str, float | bool]:
        """Report determinant, condition number and numerical solvability."""
        determinant = float(np.linalg.det(self._matrix))
        condition_number = float(np.linalg.cond(self._matrix))
        return {
            "determinant": determinant,
            "condition_number": condition_number,
            "is_well_posed": bool(
                np.isfinite(condition_number)
                and abs(determinant) > np.finfo(float).eps
                and condition_number < 1e12
            ),
        }

    def _validate_demand(self, demand: FloatArray) -> FloatArray:
        values = np.asarray(demand, dtype=np.float64)
        if values.shape != (len(self.SOURCE_NAMES),):
            raise ValueError(f"Demand must contain {len(self.SOURCE_NAMES)} values.")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Demand values must be finite and nonnegative.")
        return values

    def _solve_validated(self, demand: FloatArray) -> DaySolution:
        exact_usage = solve(self._matrix, demand, assume_a="gen", check_finite=True)
        tolerance = 1e-10
        if np.all(exact_usage >= -tolerance):
            usage = np.maximum(exact_usage, 0.0)
            residual = self._matrix @ usage - demand
            return DaySolution(usage, True, "direct", float(np.linalg.norm(residual)))

        usage, residual_norm = nnls(self._matrix, demand)
        return DaySolution(usage, False, "nnls", float(residual_norm))

    def solve_day(self, d1: float, d2: float) -> DaySolution:
        """Solve one day and switch to NNLS when an exact allocation is negative."""
        if not self.well_posedness["is_well_posed"]:
            raise np.linalg.LinAlgError("The coefficient matrix is singular or ill-conditioned.")
        demand = self._validate_demand(np.array([d1, d2], dtype=np.float64))
        return self._solve_validated(demand)

    def solve_days_loop(self, demands: FloatArray) -> list[DaySolution]:
        """Solve a batch by calling the single-day solver for each row."""
        demand_values = self._validate_demands(demands)
        return [self._solve_validated(row) for row in demand_values]

    def solve_days_vectorized(self, demands: FloatArray) -> list[DaySolution]:
        """Solve a batch with one scipy solve call and NNLS for infeasible rows."""
        if not self.well_posedness["is_well_posed"]:
            raise np.linalg.LinAlgError("The coefficient matrix is singular or ill-conditioned.")
        demand_values = self._validate_demands(demands)
        if len(demand_values) == 0:
            return []
        right_hand_side = demand_values.T
        exact_usages = solve(
            self._matrix,
            right_hand_side,
            assume_a="gen",
            check_finite=True,
        ).T
        solutions = []
        for demand, exact_usage in zip(demand_values, exact_usages):
            if np.all(exact_usage >= -1e-10):
                usage = np.maximum(exact_usage, 0.0)
                residual_norm = float(np.linalg.norm(self._matrix @ usage - demand))
                solutions.append(DaySolution(usage, True, "direct", residual_norm))
            else:
                usage, residual_norm = nnls(self._matrix, demand)
                solutions.append(DaySolution(usage, False, "nnls", float(residual_norm)))
        return solutions

    def _validate_demands(self, demands: FloatArray) -> FloatArray:
        values = np.asarray(demands, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.SOURCE_NAMES):
            raise ValueError(f"Demands must have shape (days, {len(self.SOURCE_NAMES)}).")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Demand values must be finite and nonnegative.")
        return values


class HybridMicroGrid(MicroGrid):
    """Three-source extension with solar, battery and diesel generation."""

    DEFAULT_MATRIX: FloatArray = np.array(
        [[3.0, 2.0, 1.0], [4.0, 1.0, 2.0], [1.0, 1.0, 1.0]]
    )
    SOURCE_NAMES = ("solar", "battery", "diesel")

    def solve_day(self, d1: float, d2: float, d3: float) -> DaySolution:
        """Solve the hybrid grid's three demand equations."""
        if not self.well_posedness["is_well_posed"]:
            raise np.linalg.LinAlgError("The coefficient matrix is singular or ill-conditioned.")
        demand = self._validate_demand(np.array([d1, d2, d3], dtype=np.float64))
        return self._solve_validated(demand)


def read_nonnegative_float(
    prompt: str,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], object] = print,
) -> float:
    """Prompt repeatedly until a finite, nonnegative numeric value is entered."""
    while True:
        raw_value = input_fn(prompt)
        try:
            value = float(raw_value.strip())
        except ValueError:
            output_fn("Enter a numeric value.")
            continue
        if not np.isfinite(value):
            output_fn("Enter a finite numeric value.")
            continue
        if value < 0:
            output_fn("Enter a nonnegative value.")
            continue
        return value


def generate_daily_demands(days: int = 30, seed: int = 2026) -> pd.DataFrame:
    """Generate reproducible synthetic 30-day demand with weekly variation."""
    if not isinstance(days, (int, np.integer)) or days < 1:
        raise ValueError("Days must be a positive integer.")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("Seed must be an integer.")
    rng = np.random.default_rng(seed)
    weekday_factor = np.array([1.04, 1.02, 1.0, 0.99, 1.03, 0.92, 0.88])
    factors = weekday_factor[np.arange(days) % len(weekday_factor)]
    solar_usage = np.maximum(1.0, rng.normal(42.0, 2.5, size=days) * factors)
    battery_usage = np.maximum(1.0, rng.normal(18.0, 1.8, size=days) * factors)
    d1 = 3 * solar_usage + 2 * battery_usage + rng.normal(0, 1.5, size=days)
    d2 = 4 * solar_usage + battery_usage + rng.normal(0, 1.5, size=days)
    return pd.DataFrame(
        {
            "day": np.arange(1, days + 1),
            "d1": np.maximum(d1, 0.0),
            "d2": np.maximum(d2, 0.0),
        }
    )


def load_demand_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate ``day``, ``d1`` and ``d2`` demand columns from CSV."""
    with Path(path).open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"day", "d1", "d2"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError("CSV must contain day, d1 and d2 columns.")
        records = list(reader)
    if not records:
        raise ValueError("Demand CSV must contain at least one data row.")
    try:
        frame = pd.DataFrame(
            {
                "day": [int(row["day"]) for row in records],
                "d1": [float(row["d1"]) for row in records],
                "d2": [float(row["d2"]) for row in records],
            }
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Demand CSV contains invalid numeric data.") from error
    if not np.all(np.isfinite(frame[["d1", "d2"]].to_numpy())):
        raise ValueError("Demand CSV values must be finite.")
    if (frame[["d1", "d2"]].to_numpy() < 0).any():
        raise ValueError("Demand CSV values must be nonnegative.")
    if (frame["day"] < 1).any() or frame["day"].duplicated().any():
        raise ValueError("Day values must be positive and unique.")
    return frame


def usage_statistics(solutions: list[DaySolution]) -> dict[str, dict[str, float]]:
    """Report statistics-module mean, sample variance and standard deviation."""
    if len(solutions) < 2:
        raise ValueError("At least two daily solutions are required for sample statistics.")
    source_count = len(solutions[0].usage)
    if any(len(solution.usage) != source_count for solution in solutions):
        raise ValueError("All solutions must use the same number of sources.")
    return {
        name: {
            "mean": float(mean(values)),
            "variance": float(variance(values)),
            "standard_deviation": float(stdev(values)),
        }
        for index, name in enumerate(
            ("solar", "battery", "diesel")[:source_count]
        )
        if (values := [float(solution.usage[index]) for solution in solutions])
    }


def daily_energy_cost(
    solar: FloatArray,
    battery: FloatArray,
    *,
    solar_rate: float = 150.0,
    battery_rate: float = 450.0,
) -> FloatArray:
    """Calculate daily energy cost in UGX from solar and battery kWh."""
    solar_values = np.asarray(solar, dtype=np.float64)
    battery_values = np.asarray(battery, dtype=np.float64)
    if solar_values.ndim != 1 or solar_values.shape != battery_values.shape:
        raise ValueError("Solar and battery usage must be matching one-dimensional arrays.")
    if not np.all(np.isfinite(solar_values)) or not np.all(np.isfinite(battery_values)):
        raise ValueError("Usage values must be finite.")
    if np.any(solar_values < 0) or np.any(battery_values < 0):
        raise ValueError("Usage values must be nonnegative.")
    if not np.isfinite(solar_rate) or not np.isfinite(battery_rate):
        raise ValueError("Energy rates must be finite.")
    if solar_rate < 0 or battery_rate < 0:
        raise ValueError("Energy rates must be nonnegative.")
    return solar_values * solar_rate + battery_values * battery_rate


def sensitivity_analysis(
    grid: MicroGrid,
    d1: float,
    d2: float,
    *,
    draws: int = 1_000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Measure allocation changes under independent demand perturbations of ±5%."""
    if not isinstance(draws, (int, np.integer)) or draws < 1_000:
        raise ValueError("At least 1,000 sensitivity draws are required.")
    nominal = grid._validate_demand(np.array([d1, d2], dtype=np.float64))
    if not grid.well_posedness["is_well_posed"]:
        raise np.linalg.LinAlgError("The coefficient matrix is singular or ill-conditioned.")
    rng = np.random.default_rng(seed)
    perturbations = rng.uniform(0.95, 1.05, size=(draws, 2))
    perturbed_demands = perturbations * nominal
    usage_draws = solve(grid.coefficient_matrix, perturbed_demands.T).T
    nominal_usage = solve(grid.coefficient_matrix, nominal)
    absolute_changes = np.abs(usage_draws - nominal_usage)
    return pd.DataFrame(
        {
            "source": grid.SOURCE_NAMES,
            "mean_absolute_change": absolute_changes.mean(axis=0),
            "95th_percentile_absolute_change": np.quantile(absolute_changes, 0.95, axis=0),
            "maximum_absolute_change": absolute_changes.max(axis=0),
        }
    )
