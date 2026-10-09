import pandas as pd
import pytest

from kzclimate import indices


def one_year(year: int, **columns) -> pd.DataFrame:
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D", name="date")
    base = {"tmean": 10.0, "tmax": 20.0, "tmin": 5.0, "precip": 1.0}
    base.update(columns)
    return pd.DataFrame({k: v(days) if callable(v) else v for k, v in base.items()}, index=days)


def test_counts_days_over_thresholds():
    def tmin(days):
        return [-1.0 if i < 100 else 5.0 for i in range(len(days))]

    def tmax(days):
        return [31.0 if i < 10 else (25.0 if i < 30 else 20.0) for i in range(len(days))]

    yearly = indices.annual(one_year(2001, tmin=tmin, tmax=tmax))
    row = yearly.loc[2001]
    assert row["frost_days"] == 100
    assert row["hot_days"] == 10
    assert row["summer_days"] == 10  # 25 °C exactly is not "above 25"
    assert row["ice_days"] == 0
    assert row["tropical_nights"] == 0
    assert row["precip"] == pytest.approx(365.0)


def test_leap_year_needs_all_366_days():
    full = one_year(2000)
    assert list(indices.annual(full).index) == [2000]
    assert indices.annual(full.drop(full.index[59])).empty  # 29 February missing


def test_year_with_a_missing_value_is_left_out():
    df = one_year(2001)
    df.iloc[200, 0] = float("nan")
    assert indices.annual(df).empty


def test_incomplete_current_year_is_left_out(daily):
    yearly = indices.annual(daily)
    assert yearly.index[0] == 1941  # 1940 lacks its first local day, see cities.FIRST_DATE
    assert yearly.index[-1] == 2025


def test_yearly_means_follow_the_built_in_warming(daily):
    yearly = indices.annual(daily)
    early = indices.period_mean(yearly["mean"], (1941, 1960))
    late = indices.period_mean(yearly["mean"], (2001, 2020))
    assert late - early == pytest.approx(0.03 * 60, abs=0.3)


def test_period_mean_requires_every_year(daily):
    yearly = indices.annual(daily)["mean"]
    with pytest.raises(ValueError, match="incomplete"):
        indices.period_mean(yearly.drop(1975), (1961, 1990))


def test_monthly_normals_show_the_seasonal_cycle(daily):
    normals = indices.monthly_normals(indices.monthly_means(daily), indices.NORMAL)
    assert len(normals) == 12
    assert normals.index(min(normals)) == 0  # January is the coldest
    assert normals.index(max(normals)) in (6, 7)  # July or August is the warmest


def test_partial_month_is_not_a_month(daily):
    monthly = indices.monthly_means(daily)
    assert str(monthly.index[-1]) == "2026-09"  # October has only three days


def test_recent_months_compare_with_the_normal_of_the_same_month(daily):
    monthly = indices.monthly_means(daily)
    normals = indices.monthly_normals(monthly, indices.NORMAL)
    recent = indices.recent_months(monthly, normals)
    assert [m["month"] for m in recent][:2] == ["2025-10", "2025-11"]
    assert len(recent) == 12
    for m in recent:
        month = int(m["month"][5:])
        assert m["normal"] == normals[month - 1]
        assert m["anomaly"] == pytest.approx(m["mean"] - m["normal"])
