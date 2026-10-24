"""Unit tests for the micro-grid dispatch mini-project."""

import numpy as np
import pandas as pd
import pytest

from src.microgrid import (
    HybridMicroGrid,
    MicroGrid,
    daily_energy_cost,
    generate_daily_demands,
    load_demand_csv,
    read_nonnegative_float,
    sensitivity_analysis,
    usage_statistics,
)


def test_default_matrix_is_well_posed_and_solves_day() -> None:
    grid = MicroGrid()

    solution = grid.solve_day(110, 130)

    assert grid.well_posedness["determinant"] == pytest.approx(-5)
    assert grid.well_posedness["is_well_posed"]
    assert solution.usage == pytest.approx([30, 10])
    assert solution.feasible


def test_invalid_demands_and_singular_matrix_are_rejected() -> None:
    with pytest.raises(ValueError):
        MicroGrid().solve_day(-1, 2)
    with pytest.raises(np.linalg.LinAlgError):
        MicroGrid(np.array([[1.0, 2.0], [2.0, 4.0]])).solve_day(1, 2)


def test_infeasible_solution_uses_nonnegative_least_squares() -> None:
    solution = MicroGrid().solve_day(100, 200)

    assert not solution.feasible
    assert solution.method == "nnls"
    assert np.all(solution.usage >= 0)


def test_loop_and_vectorized_solutions_match() -> None:
    demands = generate_daily_demands().loc[:, ["d1", "d2"]].to_numpy()
    grid = MicroGrid()

    loop_solutions = grid.solve_days_loop(demands)
    vectorized_solutions = grid.solve_days_vectorized(demands)

    assert len(loop_solutions) == len(vectorized_solutions) == 30
    assert all(
        np.allclose(loop.usage, vectorized.usage)
        for loop, vectorized in zip(loop_solutions, vectorized_solutions)
    )


def test_generator_is_seeded_and_loader_checks_csv(tmp_path) -> None:
    generated = generate_daily_demands(seed=5)
    assert generated.equals(generate_daily_demands(seed=5))
    assert len(generated) == 30

    csv_path = tmp_path / "demands.csv"
    generated.to_csv(csv_path, index=False)
    loaded = load_demand_csv(csv_path)
    pd.testing.assert_frame_equal(loaded, generated)


def test_interactive_input_retries_invalid_values() -> None:
    values = iter(["", "abc", "-2", "inf", "3.5"])
    messages = []

    result = read_nonnegative_float(
        "Demand: ",
        input_fn=lambda _prompt: next(values),
        output_fn=messages.append,
    )

    assert result == 3.5
    assert len(messages) == 4


def test_usage_statistics_and_cost_calculation() -> None:
    solutions = MicroGrid().solve_days_loop(np.array([[110.0, 130.0], [120.0, 140.0]]))

    stats = usage_statistics(solutions)
    costs = daily_energy_cost(
        np.array([solution.solar for solution in solutions]),
        np.array([solution.battery for solution in solutions]),
    )

    assert stats["solar"]["mean"] == pytest.approx(31)
    assert costs.tolist() == pytest.approx([9000, 10200])
    assert np.std([stats["solar"]["standard_deviation"], stats["battery"]["standard_deviation"]]) >= 0


def test_hybrid_grid_solve_and_sensitivity() -> None:
    grid = HybridMicroGrid()
    solution = grid.solve_day(100, 120, 60)
    sensitivity = sensitivity_analysis(MicroGrid(), 110, 130, draws=1_000, seed=12)

    assert solution.usage.shape == (3,)
    assert solution.diesel == pytest.approx(solution.usage[2])
    assert solution.feasible
    assert list(sensitivity["source"]) == ["solar", "battery"]
    assert len(sensitivity) == 2


def test_invalid_data_edges_are_reported() -> None:
    with pytest.raises(ValueError):
        generate_daily_demands(days=0)
    with pytest.raises(ValueError):
        usage_statistics([MicroGrid().solve_day(110, 130)])
    with pytest.raises(ValueError):
        daily_energy_cost(np.array([-1.0]), np.array([2.0]))
