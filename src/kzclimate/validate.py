"""Checks on a daily series before it is analysed. A failed check stops the release:
the dashboard keeps the previous version instead of showing numbers built on bad data."""

from __future__ import annotations

from datetime import date

import pandas as pd

from .cities import FIRST_DATE
from .fetch import COLUMNS

# Physically possible at these two places, with a wide margin: the records are about
# -52 °C and +45 °C, and the wettest days bring well under 100 mm.
TEMPERATURE_RANGE = (-65.0, 55.0)
PRECIP_RANGE = (0.0, 300.0)
# Values are rounded to 0.1, so min, mean and max may touch but never cross by more.
ORDER_TOLERANCE = 0.05
# ERA5 is about six days behind. More than this means the source stopped updating.
MAX_LAG_DAYS = 45


class DataError(ValueError):
    def __init__(self, name: str, problems: list[str]):
        super().__init__(f"{name}: " + "; ".join(problems))
        self.problems = problems


def problems(df: pd.DataFrame, *, today: date) -> list[str]:
    """Everything wrong with a daily series; empty when it is fit for analysis."""
    if df.empty:
        return ["no data"]
    found: list[str] = []

    if list(df.columns) != COLUMNS:
        return [f"columns are {list(df.columns)}, expected {COLUMNS}"]
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        found.append("dates are not strictly increasing")
    if df.index[0].date() != FIRST_DATE:
        found.append(f"series starts on {df.index[0].date()}, expected {FIRST_DATE}")

    missing = pd.date_range(df.index[0], df.index[-1], freq="D").difference(df.index)
    if len(missing):
        found.append(f"{len(missing)} days missing, the first is {missing[0].date()}")
    empty = df.isna().any(axis=1)
    if empty.any():
        found.append(f"{int(empty.sum())} days with missing values, the first is {df.index[empty][0].date()}")

    lo, hi = TEMPERATURE_RANGE
    for column in ("tmean", "tmax", "tmin"):
        bad = df[column].notna() & ~df[column].between(lo, hi)
        if bad.any():
            found.append(f"{column} outside {lo}..{hi} °C on {int(bad.sum())} days")
    lo, hi = PRECIP_RANGE
    bad = df["precip"].notna() & ~df["precip"].between(lo, hi)
    if bad.any():
        found.append(f"precip outside {lo}..{hi} mm on {int(bad.sum())} days")

    crossed = (df["tmin"] > df["tmean"] + ORDER_TOLERANCE) | (df["tmean"] > df["tmax"] + ORDER_TOLERANCE)
    if crossed.any():
        found.append(f"tmin <= tmean <= tmax broken on {int(crossed.sum())} days")

    lag = (today - df.index[-1].date()).days
    if lag > MAX_LAG_DAYS:
        found.append(f"last day is {df.index[-1].date()}, {lag} days ago: the source has stopped updating")
    return found


def check(name: str, df: pd.DataFrame, *, today: date) -> None:
    found = problems(df, today=today)
    if found:
        raise DataError(name, found)
