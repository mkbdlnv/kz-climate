import json
import urllib.parse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from kzclimate import fetch
from kzclimate.cities import CITIES, FIRST_DATE

FIXTURE = Path(__file__).parent / "fixtures" / "archive_response.json"
ASTANA = CITIES[1]


def payload_for(start: date, end: date, *, until: date | None = None, offset: float = 0.0) -> dict:
    """An archive answer for start..end: tmean is day-of-month/10 + offset, days after until are null."""
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    def values(f):
        return [round(f(d), 1) if until is None or d <= until else None for d in days]

    return {
        "daily_units": {name: unit for name, (_, unit) in fetch.VARIABLES.items()},
        "daily": {
            "time": [d.isoformat() for d in days],
            "temperature_2m_mean": values(lambda d: d.day / 10 + offset),
            "temperature_2m_max": values(lambda d: d.day / 10 + offset + 5),
            "temperature_2m_min": values(lambda d: d.day / 10 + offset - 5),
            "precipitation_sum": values(lambda d: 0.5),
        },
    }


class FakeArchive:
    """Stands in for Open-Meteo: answers from payload_for and records what was asked."""

    def __init__(self, *, until: date | None = None, offset: float = 0.0):
        self.until = until  # ERA5 has nothing after this day yet
        self.offset = offset
        self.requests: list[tuple[date, date]] = []

    def __call__(self, url: str) -> dict:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        start = date.fromisoformat(query["start_date"][0])
        end = date.fromisoformat(query["end_date"][0])
        self.requests.append((start, end))
        return payload_for(start, end, until=self.until, offset=self.offset)


def no_sleep(_seconds: float) -> None:
    pass


def test_parses_a_real_response_and_drops_days_not_yet_in_era5():
    df = fetch.parse_response(json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert list(df.columns) == fetch.COLUMNS
    assert df.index[0] == pd.Timestamp("2026-09-25")
    assert df.index[-1] == pd.Timestamp("2026-10-03")  # 4..9 October are null: ERA5 lags
    assert df.loc["2026-09-25", "precip"] == pytest.approx(4.5)
    assert not df.isna().any().any()


def test_rejects_an_unexpected_unit():
    payload = payload_for(date(2020, 1, 1), date(2020, 1, 3))
    payload["daily_units"]["temperature_2m_mean"] = "°F"
    with pytest.raises(fetch.FetchError, match="unit"):
        fetch.parse_response(payload)


def test_rejects_a_variable_of_the_wrong_length():
    payload = payload_for(date(2020, 1, 1), date(2020, 1, 3))
    payload["daily"]["precipitation_sum"].pop()
    with pytest.raises(fetch.FetchError, match="precipitation_sum"):
        fetch.parse_response(payload)


def test_rejects_an_answer_without_data():
    with pytest.raises(fetch.FetchError):
        fetch.parse_response({"error": True, "reason": "Parameter 'start_date' is out of allowed range"})


def test_url_asks_era5_only_in_local_days():
    url = fetch.build_url(ASTANA, date(2000, 1, 1), date(2000, 12, 31))
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert query["models"] == ["era5"]
    assert query["timezone"] == ["Asia/Almaty"]
    assert query["daily"][0].split(",") == list(fetch.VARIABLES)


def test_long_periods_are_downloaded_in_chunks_with_pauses():
    archive, pauses = FakeArchive(), []
    df = fetch.download(ASTANA, date(1940, 1, 1), date(1952, 3, 1), get=archive, sleep=pauses.append)
    assert archive.requests == [
        (date(1940, 1, 1), date(1944, 12, 31)),
        (date(1945, 1, 1), date(1949, 12, 31)),
        (date(1950, 1, 1), date(1952, 3, 1)),
    ]
    assert pauses == [fetch.PAUSE_SECONDS] * 2
    assert len(df) == (date(1952, 3, 1) - date(1940, 1, 1)).days + 1
    assert df.index.is_unique and df.index.is_monotonic_increasing


TODAY = date(2026, 10, 9)
ERA5_UNTIL = date(2026, 10, 3)


def test_first_run_downloads_everything_since_1940(tmp_path):
    archive = FakeArchive(until=ERA5_UNTIL)
    assert fetch.update_city(ASTANA, tmp_path, today=TODAY, get=archive, sleep=no_sleep) is True
    df = fetch.read_csv(tmp_path / "astana.csv")
    assert df.index[0].date() == FIRST_DATE
    assert df.index[-1].date() == ERA5_UNTIL
    assert len(df) == (ERA5_UNTIL - FIRST_DATE).days + 1
    assert archive.requests[0][0] == FIRST_DATE


def test_next_run_refreshes_only_the_tail(tmp_path):
    month_ago = date(2026, 9, 3)
    first = FakeArchive(until=month_ago)
    fetch.update_city(ASTANA, tmp_path, today=date(2026, 9, 9), get=first, sleep=no_sleep)
    archive = FakeArchive(until=ERA5_UNTIL)
    assert fetch.update_city(ASTANA, tmp_path, today=TODAY, get=archive, sleep=no_sleep) is True
    assert archive.requests == [(month_ago - timedelta(days=fetch.REFRESH_DAYS), TODAY)]
    df = fetch.read_csv(tmp_path / "astana.csv")
    assert df.index.is_unique
    assert df.index[-1].date() == ERA5_UNTIL
    assert len(df) == (ERA5_UNTIL - FIRST_DATE).days + 1


def test_nothing_new_leaves_the_file_unchanged(tmp_path):
    fetch.update_city(ASTANA, tmp_path, today=TODAY, get=FakeArchive(until=ERA5_UNTIL), sleep=no_sleep)
    again = fetch.update_city(ASTANA, tmp_path, today=TODAY, get=FakeArchive(until=ERA5_UNTIL), sleep=no_sleep)
    assert again is False


def test_revised_days_replace_stored_ones(tmp_path):
    fetch.update_city(ASTANA, tmp_path, today=TODAY, get=FakeArchive(until=ERA5_UNTIL), sleep=no_sleep)
    # ERA5T days are preliminary: the re-downloaded tail comes back slightly different.
    revised = FakeArchive(until=ERA5_UNTIL, offset=0.3)
    assert fetch.update_city(ASTANA, tmp_path, today=TODAY, get=revised, sleep=no_sleep) is True
    df = fetch.read_csv(tmp_path / "astana.csv")
    assert df.loc["2026-10-01", "tmean"] == pytest.approx(0.1 + 0.3)
    assert df.loc["2000-01-01", "tmean"] == pytest.approx(0.1)  # outside the refreshed tail


def test_csv_round_trip_keeps_values_and_dates(tmp_path):
    df = fetch.parse_response(json.loads(FIXTURE.read_text(encoding="utf-8")))
    path = tmp_path / "x.csv"
    fetch.write_csv(df, path)
    assert path.read_text(encoding="utf-8").splitlines()[:2] == [
        "date,tmean,tmax,tmin,precip",
        "2026-09-25,12.2,13.8,10.3,4.5",
    ]
    back = fetch.read_csv(path)
    assert list(back.index) == list(df.index)
    assert list(back.columns) == list(df.columns)
    assert (back.to_numpy() == df.to_numpy()).all()
