"""Rainfall summaries, crop-screening rules, and regional comparisons."""

from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from statistics import mean, pstdev
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine

FloatArray = NDArray[np.float64]
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


@dataclass(frozen=True)
class Region:
    """One region's 12 monthly rainfall normals or illustrative totals in mm."""

    name: str
    rainfall_mm: FloatArray

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Region name must not be empty.")
        rainfall = np.asarray(self.rainfall_mm, dtype=np.float64).copy()
        if rainfall.shape != (12,):
            raise ValueError("Rainfall must contain exactly 12 monthly values.")
        if not np.all(np.isfinite(rainfall)) or np.any(rainfall < 0):
            raise ValueError("Rainfall values must be finite and nonnegative.")
        rainfall.setflags(write=False)
        object.__setattr__(self, "rainfall_mm", rainfall)

    def annual_total(self) -> float:
        """Return annual rainfall total in millimetres."""
        return float(np.sum(self.rainfall_mm))

    def monthly_mean(self) -> float:
        """Return mean monthly rainfall in millimetres."""
        return float(mean(self.rainfall_mm.tolist()))

    def wettest_month(self) -> tuple[str, float]:
        """Return the name and rainfall of the wettest month."""
        index = int(np.argmax(self.rainfall_mm))
        return MONTH_NAMES[index], float(self.rainfall_mm[index])

    def driest_month(self) -> tuple[str, float]:
        """Return the name and rainfall of the driest month."""
        index = int(np.argmin(self.rainfall_mm))
        return MONTH_NAMES[index], float(self.rainfall_mm[index])

    def coefficient_of_variation(self) -> float:
        """Return population standard deviation divided by monthly mean."""
        average = self.monthly_mean()
        if average == 0:
            raise ValueError("Coefficient of variation is undefined for zero mean rainfall.")
        return float(pstdev(self.rainfall_mm.tolist()) / average)


@dataclass(frozen=True)
class CropRule:
    """A monthly rainfall screening range for one crop, in millimetres."""

    crop_name: str
    minimum_mm: float
    maximum_mm: float

    def __post_init__(self) -> None:
        if not self.crop_name.strip():
            raise ValueError("Crop name must not be empty.")
        if not np.isfinite(self.minimum_mm) or not np.isfinite(self.maximum_mm):
            raise ValueError("Crop rainfall thresholds must be finite.")
        if self.minimum_mm < 0 or self.maximum_mm <= self.minimum_mm:
            raise ValueError("Crop rainfall range must be nonnegative and increasing.")

    def classify(self, rainfall_mm: float) -> str:
        """Classify a month as suitable, drought-screened, or excess-rainfall."""
        if not np.isfinite(rainfall_mm) or rainfall_mm < 0:
            raise ValueError("Monthly rainfall must be finite and nonnegative.")
        if rainfall_mm < self.minimum_mm:
            return "Drought risk"
        if rainfall_mm > self.maximum_mm:
            return "Waterlogging risk"
        return f"Good for {self.crop_name}"


def cosine_similarity(first: FloatArray, second: FloatArray) -> float:
    """Calculate vector cosine similarity using the dot product and norms."""
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if (
        first_values.ndim != 1
        or second_values.ndim != 1
        or first_values.shape != second_values.shape
        or len(first_values) == 0
    ):
        raise ValueError("Cosine similarity needs matching nonempty one-dimensional vectors.")
    if not np.all(np.isfinite(first_values)) or not np.all(np.isfinite(second_values)):
        raise ValueError("Cosine similarity vectors must be finite.")
    denominator = float(np.linalg.norm(first_values) * np.linalg.norm(second_values))
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return float(np.dot(first_values, second_values) / denominator)


def region_comparison_matrices(
    regions: dict[str, Region],
) -> dict[str, pd.DataFrame]:
    """Return pairwise cosine similarity, Pearson correlation and Euclidean distance."""
    if len(regions) < 2:
        raise ValueError("At least two regions are needed for pairwise comparison.")
    names = list(regions)
    vectors = [regions[name].rainfall_mm for name in names]
    cosine_matrix = np.empty((len(names), len(names)), dtype=np.float64)
    pearson_matrix = np.empty_like(cosine_matrix)
    distance_matrix = np.empty_like(cosine_matrix)
    for row, first in enumerate(vectors):
        if np.std(first) == 0:
            raise ValueError(f"Pearson correlation is undefined for constant rainfall in {names[row]}.")
        for column, second in enumerate(vectors):
            cosine_matrix[row, column] = cosine_similarity(first, second)
            if row == column:
                pearson_matrix[row, column] = 1.0
                distance_matrix[row, column] = 0.0
            else:
                if np.std(second) == 0:
                    raise ValueError(f"Pearson correlation is undefined for constant rainfall in {names[column]}.")
                pearson_matrix[row, column] = float(np.corrcoef(first, second)[0, 1])
                distance_matrix[row, column] = float(np.linalg.norm(first - second))
    return {
        "Cosine similarity": pd.DataFrame(cosine_matrix, index=names, columns=names),
        "Pearson correlation": pd.DataFrame(pearson_matrix, index=names, columns=names),
        "Euclidean distance (mm)": pd.DataFrame(distance_matrix, index=names, columns=names),
    }


def scipy_cosine_similarity(first: FloatArray, second: FloatArray) -> float:
    """Compute cosine similarity using SciPy cosine distance for verification."""
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if (
        first_values.ndim != 1
        or second_values.ndim != 1
        or first_values.shape != second_values.shape
        or len(first_values) == 0
    ):
        raise ValueError("Cosine similarity needs matching nonempty one-dimensional vectors.")
    if not np.all(np.isfinite(first_values)) or not np.all(np.isfinite(second_values)):
        raise ValueError("Cosine similarity vectors must be finite.")
    if np.linalg.norm(first_values) == 0 or np.linalg.norm(second_values) == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return float(1 - cosine(first_values, second_values))


def detect_rainy_seasons(
    region: Region,
    *,
    minimum_prominence_mm: float = 20.0,
    minimum_peak_distance_months: int = 2,
) -> dict[str, object]:
    """Detect distinct seasonal peaks on a circular January–December series.

    The series is repeated three times and peaks are collected only from the
    middle year, so December/January seasons are handled without duplicate
    boundary peaks. Returned indices are zero-based calendar months.
    Peaks must meet both the explicit prominence and separation criteria.
    """
    if not np.isfinite(minimum_prominence_mm) or minimum_prominence_mm < 0:
        raise ValueError("Minimum peak prominence must be finite and nonnegative.")
    if (
        not isinstance(minimum_peak_distance_months, (int, np.integer))
        or minimum_peak_distance_months < 1
    ):
        raise ValueError("Minimum peak distance must be a positive integer.")
    values = region.rainfall_mm
    padded = np.tile(values, 3)
    peaks, properties = find_peaks(
        padded,
        prominence=minimum_prominence_mm,
        distance=minimum_peak_distance_months,
    )
    month_indices = sorted(
        int(index - 12) % 12
        for index in peaks
        if 12 <= index < 24
    )
    count = len(month_indices)
    if count == 0:
        classification = "No clear peak"
    elif count == 1:
        classification = "Unimodal"
    elif count == 2:
        classification = "Bimodal"
    else:
        classification = "Multimodal"
    prominences = {
        int(index - 12) % 12: float(prominence)
        for index, prominence in zip(peaks, properties["prominences"])
        if 12 <= index < 24
    }
    return {
        "classification": classification,
        "peak_month_indices": month_indices,
        "peak_months": [MONTH_NAMES[index] for index in month_indices],
        "peak_prominences_mm": prominences,
    }


def classify_crop_suitability(
    regions: dict[str, Region],
    crop_rules: list[CropRule],
) -> pd.DataFrame:
    """Classify each region/month/crop combination with a CropRule."""
    if not regions or not crop_rules:
        raise ValueError("At least one region and one crop rule are required.")
    rows = [
        {
            "region": region_name,
            "month": month,
            "crop": rule.crop_name,
            "rainfall_mm": float(region.rainfall_mm[index]),
            "classification": rule.classify(float(region.rainfall_mm[index])),
        }
        for region_name, region in regions.items()
        for index, month in enumerate(MONTH_NAMES)
        for rule in crop_rules
    ]
    return pd.DataFrame(rows)


def fetch_nasa_power_monthly_rainfall(
    latitude: float,
    longitude: float,
    start_year: int,
    end_year: int,
    *,
    timeout_seconds: float = 60.0,
) -> pd.DataFrame:
    """Fetch NASA POWER corrected precipitation and return monthly mm totals.

    NASA POWER monthly ``PRECTOTCORR`` is a daily precipitation rate in
    mm/day averaged over each month. Multiplying by calendar month length
    produces an estimated monthly accumulation in mm.
    """
    if not np.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("Latitude must be finite and between -90 and 90.")
    if not np.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("Longitude must be finite and between -180 and 180.")
    if (
        not isinstance(start_year, (int, np.integer))
        or not isinstance(end_year, (int, np.integer))
        or start_year < 1981
        or end_year < start_year
    ):
        raise ValueError("Years must be integers with 1981 <= start_year <= end_year.")
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Timeout must be finite and positive.")
    query = urlencode(
        {
            "parameters": "PRECTOTCORR",
            "community": "AG",
            "longitude": longitude,
            "latitude": latitude,
            "start": start_year,
            "end": end_year,
            "format": "JSON",
        }
    )
    url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?{query}"
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to retrieve NASA POWER rainfall data: {error}") from error
    try:
        monthly_rates = payload["properties"]["parameter"]["PRECTOTCORR"]
    except (KeyError, TypeError) as error:
        raise ValueError("NASA POWER response did not contain PRECTOTCORR data.") from error

    records = []
    for period, daily_rate in monthly_rates.items():
        year, month = int(period[:4]), int(period[4:])
        if year < start_year or year > end_year or not 1 <= month <= 12:
            continue
        value = float(daily_rate)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"NASA POWER returned an invalid rainfall rate for {period}.")
        records.append(
            {
                "year": year,
                "month": month,
                "rainfall_mm": value * calendar.monthrange(year, month)[1],
            }
        )
    frame = pd.DataFrame(records, columns=("year", "month", "rainfall_mm"))
    expected_months = (end_year - start_year + 1) * 12
    if len(frame) != expected_months:
        raise ValueError(
            f"NASA POWER returned {len(frame)} months; expected {expected_months}."
        )
    return frame.sort_values(["year", "month"], ignore_index=True)
