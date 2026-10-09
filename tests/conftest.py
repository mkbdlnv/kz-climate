"""Synthetic daily series with a known trend, so tests know what the analysis must find."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kzclimate.fetch import COLUMNS


def synthetic_daily(
    start: str = "1940-01-02",
    end: str = "2025-12-31",
    *,
    trend_per_year: float = 0.03,
    seed: int = 1,
) -> pd.DataFrame:
    """Seasonal cycle (coldest in mid-January) + linear warming + weather noise, rounded like ERA5."""
    days = pd.date_range(start, end, freq="D", name="date")
    years = (days - pd.Timestamp("1940-01-01")).days.to_numpy() / 365.25
    rng = np.random.default_rng(seed)
    season = -12 * np.cos(2 * np.pi * (days.dayofyear.to_numpy() - 15) / 365.25)
    tmean = 8 + season + trend_per_year * years + rng.normal(0, 3, len(days))
    spread = 5 + 3 * rng.random(len(days))
    df = pd.DataFrame(
        {
            "tmean": tmean.round(1),
            "tmax": (tmean + spread).round(1),
            "tmin": (tmean - spread).round(1),
            "precip": rng.gamma(0.4, 4, len(days)).round(1),
        },
        index=days,
    )
    return df[COLUMNS]


@pytest.fixture(scope="session")
def daily() -> pd.DataFrame:
    # Ends in October: the last, incomplete year must be left out of yearly statistics.
    return synthetic_daily(end="2026-10-03")
