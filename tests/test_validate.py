from datetime import date, timedelta

import pytest

from kzclimate import validate

FRESH = date(2026, 10, 9)  # six days after the synthetic series ends, like real ERA5


def test_clean_series_passes(daily):
    assert validate.problems(daily, today=FRESH) == []
    validate.check("x", daily, today=FRESH)


def test_missing_day_is_reported(daily):
    found = validate.problems(daily.drop(daily.index[1000]), today=FRESH)
    assert any("1 days missing" in p for p in found)


def test_missing_value_is_reported(daily):
    df = daily.copy()
    df.iloc[5, 3] = float("nan")
    assert any("missing values" in p for p in validate.problems(df, today=FRESH))


def test_impossible_temperature_is_reported(daily):
    df = daily.copy()
    df.iloc[10, 1] = 99.0
    assert any("tmax outside" in p for p in validate.problems(df, today=FRESH))


def test_negative_precipitation_is_reported(daily):
    df = daily.copy()
    df.iloc[10, 3] = -1.0
    assert any("precip outside" in p for p in validate.problems(df, today=FRESH))


def test_min_above_mean_is_reported(daily):
    df = daily.copy()
    df.iloc[10, 2] = df.iloc[10, 0] + 1
    assert any("tmin <= tmean <= tmax" in p for p in validate.problems(df, today=FRESH))


def test_series_must_start_in_1940(daily):
    assert any("starts on" in p for p in validate.problems(daily.iloc[365:], today=FRESH))


def test_stale_source_is_reported(daily):
    late = FRESH + timedelta(days=validate.MAX_LAG_DAYS)
    assert any("stopped updating" in p for p in validate.problems(daily, today=late))


def test_wrong_columns_stop_the_other_checks(daily):
    assert validate.problems(daily.rename(columns={"precip": "rain"}), today=FRESH) == [
        "columns are ['tmean', 'tmax', 'tmin', 'rain'], expected ['tmean', 'tmax', 'tmin', 'precip']"
    ]


def test_check_raises_with_every_problem(daily):
    df = daily.copy()
    df.iloc[10, 1] = 99.0
    df.iloc[11, 3] = -1.0
    with pytest.raises(validate.DataError) as e:
        validate.check("astana", df, today=FRESH)
    assert len(e.value.problems) == 2
    assert str(e.value).startswith("astana: ")
