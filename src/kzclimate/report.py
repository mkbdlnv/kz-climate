"""Assemble climate.json, the file a release publishes and the dashboard reads.

The format is described in docs/data-format.md. A change that breaks readers bumps SCHEMA.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from . import indices
from .cities import City
from .trend import trend

SCHEMA = 1
REPOSITORY = "https://github.com/mkbdlnv/kz-climate"

SOURCE = {
    "dataset": "ERA5",
    "provider": "Open-Meteo",
    "url": "https://open-meteo.com/en/docs/historical-weather-api",
    "license": "CC BY 4.0",
    "citation": (
        "Hersbach, H. et al. (2023): ERA5 hourly data on single levels from 1940 to present. "
        "Copernicus Climate Change Service (C3S) Climate Data Store. Weather data by Open-Meteo.com"
    ),
}

# Yearly series whose trend is reported. Ice days and tropical nights are kept in the
# yearly rows only: in one of the two cities they are zero almost every year.
TREND_SERIES = ("mean", "tmax_mean", "tmin_mean", "precip", "frost_days", "summer_days", "hot_days")
COUNT_COLUMNS = tuple(indices.INDEX_LABELS)


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _rounded(d: dict, digits: int) -> dict:
    return {k: _round(v, digits) if isinstance(v, float) else v for k, v in d.items()}


def _trend(years, values) -> dict:
    t = trend(years, values)
    return {
        **t,
        "per_decade": _round(t["per_decade"], 3),
        "ci": [_round(v, 3) for v in t["ci"]],
        "change": _round(t["change"]),
        # Significant figures, not decimals: p-values of long warming series are tiny.
        "p_value": float(f"{t['p_value']:.3g}"),
        "tau": _round(t["tau"], 3),
    }


def city_report(city: City, df: pd.DataFrame) -> dict:
    yearly = indices.annual(df)
    baseline = indices.period_mean(yearly["mean"], indices.BASELINE)
    anomaly = yearly["mean"] - baseline
    monthly = indices.monthly_means(df)
    normal = indices.monthly_normals(monthly, indices.NORMAL)
    years = yearly.index.to_numpy()

    rank = anomaly.rank(ascending=False, method="min")
    last_year = int(years[-1])
    warmest = anomaly.sort_values(ascending=False).head(5)

    rows = []
    for year, row in yearly.iterrows():
        rows.append(
            {
                "year": int(year),
                "mean": _round(row["mean"]),
                "anomaly": _round(anomaly[year]),
                "tmax_mean": _round(row["tmax_mean"]),
                "tmin_mean": _round(row["tmin_mean"]),
                "precip": _round(row["precip"], 1),
                **{column: int(row[column]) for column in COUNT_COLUMNS},
            }
        )

    return {
        "id": city.id,
        "name": city.name,
        "lat": city.lat,
        "lon": city.lon,
        "data_from": df.index[0].date().isoformat(),
        "data_through": df.index[-1].date().isoformat(),
        "baseline_mean": _round(baseline),
        "years": rows,
        "trends": {name: _trend(years, yearly[name].to_numpy()) for name in TREND_SERIES},
        "warmest": [{"year": int(y), "anomaly": _round(a)} for y, a in warmest.items()],
        "last_year": {"year": last_year, "rank": int(rank[last_year]), "of": len(years)},
        "normals": [_round(v) for v in normal],
        "recent_months": [_rounded(m, 2) for m in indices.recent_months(monthly, normal)],
    }


def build(
    data: dict[City, pd.DataFrame],
    *,
    version: str,
    commit: str | None = None,
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now(UTC)
    return {
        "schema": SCHEMA,
        "version": version,
        "generated_at": now.isoformat(timespec="seconds"),
        "source": SOURCE,
        "code": {"repository": REPOSITORY, "commit": commit},
        "baseline": list(indices.BASELINE),
        "normal": list(indices.NORMAL),
        "indices": indices.INDEX_LABELS,
        "cities": [city_report(city, df) for city, df in data.items()],
    }
