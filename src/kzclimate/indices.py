"""Yearly and monthly aggregates of the daily series: means, anomalies, extreme-day counts, normals."""

from __future__ import annotations

import calendar

import pandas as pd

# Anomalies are counted from 1961–1990: the WMO reference period for climate-change
# assessments, also used by warming stripes. Recent months are compared with 1991–2020,
# the current WMO standard normal, which is what a weather service calls "the norm".
BASELINE = (1961, 1990)
NORMAL = (1991, 2020)

# Day counts per year. Thresholds follow the ETCCDI indices (FD, ID, SU, TR); hot_days
# (TX >= 30 °C) is not an ETCCDI index but is the threshold a reader expects for "hot".
INDEX_LABELS = {
    "frost_days": "дни с морозом (Tmin < 0 °C), ETCCDI FD",
    "ice_days": "дни без оттепели (Tmax < 0 °C), ETCCDI ID",
    "summer_days": "летние дни (Tmax > 25 °C), ETCCDI SU",
    "hot_days": "жаркие дни (Tmax ≥ 30 °C)",
    "tropical_nights": "тропические ночи (Tmin > 20 °C), ETCCDI TR",
}


def annual(df: pd.DataFrame) -> pd.DataFrame:
    """One row per complete calendar year: mean temperatures, precipitation total and day counts.

    A year is complete when every day has every value; the current year never is, so
    it does not enter trends with a few months standing in for twelve.
    """
    full = df.dropna()
    year = full.index.year
    by_year = full.groupby(year)
    out = pd.DataFrame(
        {
            "days": by_year.size(),
            "mean": by_year["tmean"].mean(),
            "tmax_mean": by_year["tmax"].mean(),
            "tmin_mean": by_year["tmin"].mean(),
            "precip": by_year["precip"].sum(),
            "frost_days": (full["tmin"] < 0).groupby(year).sum(),
            "ice_days": (full["tmax"] < 0).groupby(year).sum(),
            "summer_days": (full["tmax"] > 25).groupby(year).sum(),
            "hot_days": (full["tmax"] >= 30).groupby(year).sum(),
            "tropical_nights": (full["tmin"] > 20).groupby(year).sum(),
        }
    )
    days_in_year = [366 if calendar.isleap(y) else 365 for y in out.index]
    out = out[out["days"] == days_in_year].drop(columns="days")
    out.index.name = "year"
    return out


def period_mean(yearly: pd.Series, period: tuple[int, int]) -> float:
    """Mean of yearly values over period; every year of it must be there."""
    start, end = period
    window = yearly.loc[start:end]
    if len(window) != end - start + 1:
        raise ValueError(f"period {start}–{end} is incomplete: {len(window)} years")
    return float(window.mean())


def monthly_means(df: pd.DataFrame) -> pd.Series:
    """Mean daily temperature of every complete month, indexed by month (Period)."""
    tmean = df["tmean"].dropna()
    by_month = tmean.groupby(tmean.index.to_period("M"))
    means = by_month.mean()
    complete = by_month.size().to_numpy() == means.index.days_in_month.to_numpy()
    return means[complete]


def monthly_normals(monthly: pd.Series, period: tuple[int, int]) -> list[float]:
    """Twelve values: the mean temperature of each calendar month over period."""
    start, end = period
    window = monthly[(monthly.index.year >= start) & (monthly.index.year <= end)]
    by_month = window.groupby(window.index.month)
    sizes = by_month.size()
    years = end - start + 1
    if len(sizes) != 12 or (sizes != years).any():
        raise ValueError(f"normal {start}–{end} needs all 12 months of all {years} years")
    means = by_month.mean()
    return [float(means[m]) for m in range(1, 13)]


def recent_months(monthly: pd.Series, normals: list[float], count: int = 12) -> list[dict]:
    """The last complete months with their deviation from the normal of the same month."""
    out = []
    for month, value in monthly.tail(count).items():
        normal = normals[month.month - 1]
        out.append({"month": str(month), "mean": float(value), "normal": normal, "anomaly": float(value) - normal})
    return out
