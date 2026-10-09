"""Trend of a yearly series: the Mann–Kendall test says whether there is one, Sen's slope how steep.

Both are rank-based, so they need no normality and are not thrown by a single extreme
year, which is why climatology uses them for station and reanalysis series. Their
limitation: they assume independent years; serial correlation of yearly means makes
the p-value somewhat optimistic (see README).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


@dataclass(frozen=True)
class MannKendall:
    s: int  # sum of signs of all later-minus-earlier differences
    var_s: float
    z: float
    p: float  # two-sided
    tau: float  # Kendall's tau between time and value, -1..1


def mann_kendall(values: ArrayLike, *, continuity: bool = True) -> MannKendall:
    """Mann–Kendall trend test with the tie correction of the variance.

    values must be in time order and evenly spaced (one per year here).
    continuity applies the usual ±1 correction to S before the normal approximation.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 3:
        raise ValueError("need at least 3 values")
    if np.isnan(x).any():
        raise ValueError("series contains NaN")

    i, j = np.triu_indices(n, k=1)
    s = int(np.sign(x[j] - x[i]).sum())

    _, counts = np.unique(x, return_counts=True)
    ties = counts[counts > 1].astype(float)
    var_s = (n * (n - 1) * (2 * n + 5) - float(np.sum(ties * (ties - 1) * (2 * ties + 5)))) / 18

    if var_s == 0:  # every value the same: no evidence of anything
        return MannKendall(s=0, var_s=0.0, z=0.0, p=1.0, tau=0.0)
    numerator = s - math.copysign(1, s) if continuity and s != 0 else s
    z = numerator / math.sqrt(var_s)
    p = float(2 * stats.norm.sf(abs(z)))
    return MannKendall(s=s, var_s=var_s, z=z, p=p, tau=s / (n * (n - 1) / 2))


@dataclass(frozen=True)
class SenSlope:
    slope: float  # change per unit of x (per year here)
    intercept: float
    low: float  # confidence interval of the slope
    high: float


def sen_slope(x: ArrayLike, values: ArrayLike, *, confidence: float = 0.95) -> SenSlope:
    """Theil–Sen estimator: the median of the slopes between every pair of points.

    scipy's implementation is used rather than a hand-written one: it is tested
    and gives the confidence interval of Sen (1968).
    """
    result = stats.theilslopes(values, x, alpha=confidence)
    return SenSlope(
        slope=float(result.slope),
        intercept=float(result.intercept),
        low=float(result.low_slope),
        high=float(result.high_slope),
    )


def trend(years: ArrayLike, values: ArrayLike, *, alpha: float = 0.05) -> dict:
    """Trend summary for climate.json: change per decade with its 95 % interval and significance."""
    years = np.asarray(years, dtype=float)
    mk = mann_kendall(values)
    sen = sen_slope(years, values)
    span = years[-1] - years[0]
    return {
        "per_decade": sen.slope * 10,
        "ci": [sen.low * 10, sen.high * 10],
        "change": sen.slope * span,
        "p_value": mk.p,
        "tau": mk.tau,
        "significant": mk.p < alpha,
        "from": int(years[0]),
        "to": int(years[-1]),
    }
