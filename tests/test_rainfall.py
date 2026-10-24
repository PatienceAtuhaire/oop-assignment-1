"""Unit tests for rainfall-pattern and crop-suitability analysis."""

import numpy as np
import pytest

from src.rainfall import (
    CropRule,
    Region,
    classify_crop_suitability,
    cosine_similarity,
    detect_rainy_seasons,
    region_comparison_matrices,
    scipy_cosine_similarity,
)


def test_region_summaries_report_months_totals_and_variability() -> None:
    region = Region("Test", np.arange(1, 13, dtype=float))

    assert region.annual_total() == 78
    assert region.monthly_mean() == 6.5
    assert region.wettest_month() == ("December", 12.0)
    assert region.driest_month() == ("January", 1.0)
    assert region.coefficient_of_variation() > 0


@pytest.mark.parametrize(
    "rainfall",
    [np.ones(11), np.array([-1.0] + [1.0] * 11), np.full(12, np.inf)],
)
def test_region_rejects_wrong_length_or_invalid_rainfall(rainfall) -> None:
    with pytest.raises(ValueError):
        Region("Invalid", rainfall)


def test_crop_rule_classifies_drought_suitable_and_excess_months() -> None:
    maize = CropRule("maize", 80, 200)

    assert maize.classify(40) == "Drought risk"
    assert maize.classify(120) == "Good for maize"
    assert maize.classify(240) == "Waterlogging risk"


def test_crop_suitability_covers_every_month_region_and_crop() -> None:
    regions = {
        "A": Region("A", np.full(12, 100.0)),
        "B": Region("B", np.full(12, 250.0)),
    }
    rules = [CropRule("maize", 80, 200), CropRule("beans", 50, 150)]

    classifications = classify_crop_suitability(regions, rules)

    assert len(classifications) == 2 * 12 * 2
    assert set(classifications["region"]) == {"A", "B"}
    assert set(classifications["crop"]) == {"maize", "beans"}
    assert (classifications["region"] == "A").sum() == 24


def test_cosine_similarity_matches_scipy_and_is_scale_invariant() -> None:
    first = np.array([1.0, 2.0, 3.0])
    second = 8 * first

    similarity = cosine_similarity(first, second)

    assert similarity == pytest.approx(1.0)
    assert similarity == pytest.approx(scipy_cosine_similarity(first, second))


def test_cosine_similarity_rejects_zero_or_mismatched_vectors() -> None:
    with pytest.raises(ValueError):
        cosine_similarity(np.zeros(3), np.ones(3))
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(3), np.ones(2))


def test_region_matrices_have_expected_symmetry_and_diagonals() -> None:
    regions = {
        "A": Region("A", np.array([1, 2, 4, 6, 8, 10, 8, 6, 4, 3, 2, 1], dtype=float)),
        "B": Region("B", np.array([2, 4, 8, 12, 16, 20, 16, 12, 8, 6, 4, 2], dtype=float)),
        "C": Region("C", np.arange(12, 0, -1, dtype=float)),
    }

    matrices = region_comparison_matrices(regions)
    cosine_matrix = matrices["Cosine similarity"].to_numpy()
    pearson_matrix = matrices["Pearson correlation"].to_numpy()
    distance_matrix = matrices["Euclidean distance (mm)"].to_numpy()

    assert np.allclose(np.diag(cosine_matrix), 1)
    assert np.allclose(np.diag(pearson_matrix), 1)
    assert np.allclose(np.diag(distance_matrix), 0)
    assert np.allclose(distance_matrix, distance_matrix.T)
    assert cosine_matrix[0, 1] == pytest.approx(1.0)
    assert distance_matrix[0, 1] > 0


def test_peak_finder_identifies_unimodal_and_bimodal_patterns() -> None:
    unimodal = Region("Unimodal", np.array([10, 20, 40, 100, 180, 100, 40, 20, 10, 8, 7, 8], dtype=float))
    bimodal = Region("Bimodal", np.array([10, 20, 80, 160, 100, 25, 20, 80, 170, 90, 25, 10], dtype=float))

    assert detect_rainy_seasons(unimodal)["classification"] == "Unimodal"
    assert detect_rainy_seasons(bimodal)["classification"] == "Bimodal"
