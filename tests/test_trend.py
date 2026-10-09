import numpy as np
import pytest
from scipy import stats

from kzclimate.trend import mann_kendall, sen_slope, trend


def test_strictly_increasing_series_is_a_certain_trend():
    mk = mann_kendall(range(20))
    assert mk.s == 20 * 19 // 2
    assert mk.tau == 1
    assert mk.p < 1e-6


def test_decreasing_series_has_negative_statistic():
    mk = mann_kendall([5, 4, 3, 2, 1, 0])
    assert mk.s == -15
    assert mk.tau == -1


def test_constant_series_has_no_trend():
    mk = mann_kendall([5.0] * 10)
    assert (mk.s, mk.z, mk.p, mk.tau) == (0, 0.0, 1.0, 0.0)


def test_matches_kendall_tau_without_ties():
    # Without ties and without the continuity correction, Mann–Kendall is Kendall's tau
    # between time and value, so scipy's asymptotic test must give the same numbers.
    rng = np.random.default_rng(7)
    x = rng.normal(size=60) + np.linspace(0, 1, 60)
    mk = mann_kendall(x, continuity=False)
    tau, p = stats.kendalltau(np.arange(60), x, method="asymptotic")
    assert mk.tau == pytest.approx(tau)
    assert mk.p == pytest.approx(p, rel=1e-9)


def test_continuity_correction_makes_the_test_more_conservative():
    x = np.random.default_rng(3).normal(size=30) + np.linspace(0, 1, 30)
    assert mann_kendall(x).p > mann_kendall(x, continuity=False).p


def test_ties_reduce_the_variance():
    n = 6
    no_ties = n * (n - 1) * (2 * n + 5) / 18
    # Three pairs of ties, each removes 2*1*9 / 18 = 1.
    assert mann_kendall([1, 1, 2, 2, 3, 3]).var_s == pytest.approx(no_ties - 3)


@pytest.mark.parametrize("bad", [[1.0, 2.0], [1.0, np.nan, 3.0, 4.0]])
def test_rejects_short_or_incomplete_series(bad):
    with pytest.raises(ValueError):
        mann_kendall(bad)


def test_sen_slope_of_a_line_is_its_slope():
    years = np.arange(1940, 2026)
    sen = sen_slope(years, 0.03 * years - 50)
    assert sen.slope == pytest.approx(0.03)
    assert sen.low <= sen.slope <= sen.high


def test_sen_slope_ignores_an_outlier():
    years = np.arange(1940, 2026)
    values = 0.03 * years
    values[-1] += 100  # one absurd year moves a least-squares fit, not the median slope
    assert sen_slope(years, values).slope == pytest.approx(0.03)
    assert stats.linregress(years, values).slope > 0.04


def test_trend_summary_of_a_warming_series():
    years = np.arange(1940, 2026)
    values = 0.02 * (years - 1940) + np.random.default_rng(1).normal(0, 0.2, len(years))
    t = trend(years, values)
    assert t["per_decade"] == pytest.approx(0.2, abs=0.03)
    assert t["ci"][0] < t["per_decade"] < t["ci"][1]
    assert t["change"] == pytest.approx(t["per_decade"] / 10 * 85)
    assert t["significant"] is True
    assert (t["from"], t["to"]) == (1940, 2025)


def test_warming_then_cooling_is_not_a_monotonic_trend():
    # A symmetric V: every rising pair has a mirrored falling one, so S is exactly 0.
    years = np.arange(1940, 2026)
    t = trend(years, np.abs(years - 1982.5))
    assert t["p_value"] == 1.0
    assert t["significant"] is False
