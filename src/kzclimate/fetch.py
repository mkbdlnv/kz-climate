"""Download daily ERA5 values from the Open-Meteo archive and keep data/raw/<city>.csv current.

The first run downloads everything since 1940 in chunks; later runs re-download only the
last REFRESH_DAYS days, because the newest ERA5 days are preliminary (ERA5T) and can be
revised, and append what is new.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .cities import FIRST_DATE, TIMEZONE, City

API_URL = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = "kz-climate/1.0 (+https://github.com/mkbdlnv/kz-climate)"

# Open-Meteo daily variable -> column in data/raw/<city>.csv, with the unit we expect.
VARIABLES: dict[str, tuple[str, str]] = {
    "temperature_2m_mean": ("tmean", "°C"),
    "temperature_2m_max": ("tmax", "°C"),
    "temperature_2m_min": ("tmin", "°C"),
    "precipitation_sum": ("precip", "mm"),
}
COLUMNS = [column for column, _ in VARIABLES.values()]

REFRESH_DAYS = 120
# Open-Meteo counts a long request as many calls (one per two weeks of data) and allows
# 600 calls a minute. Five years is ~130 calls; the pause keeps a full download under the limit.
CHUNK_YEARS = 5
PAUSE_SECONDS = 15.0

GetJSON = Callable[[str], dict[str, Any]]


class FetchError(RuntimeError):
    """Open-Meteo did not answer, or answered with something we cannot use."""


def build_url(city: City, start: date, end: date) -> str:
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(VARIABLES),
        "timezone": TIMEZONE,
        # One model for the whole period: the default "best match" mixes in other
        # reanalyses for recent years, which would put a seam into the series.
        "models": "era5",
    }
    return f"{API_URL}?{urllib.parse.urlencode(params)}"


def get_json(url: str, *, retries: int = 4, timeout: float = 120) -> dict[str, Any]:
    """GET url and decode JSON; rate limits and server errors are retried with a growing pause."""
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            retriable = e.code == 429 or e.code >= 500
            if not retriable or attempt == retries - 1:
                body = e.read()[:300].decode("utf-8", "replace")
                raise FetchError(f"Open-Meteo answered HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise FetchError(f"Open-Meteo is unreachable: {e}") from e
        time.sleep(30 * (attempt + 1))
    raise AssertionError("unreachable")


def parse_response(payload: dict[str, Any]) -> pd.DataFrame:
    """Turn an archive response into a frame indexed by date, one column per variable."""
    daily = payload.get("daily")
    if not isinstance(daily, dict) or "time" not in daily:
        raise FetchError("response has no daily data")
    units = payload.get("daily_units") or {}
    days = daily["time"]
    columns: dict[str, list[float | None]] = {}
    for name, (column, unit) in VARIABLES.items():
        if units.get(name) != unit:
            raise FetchError(f"{name}: expected unit {unit!r}, got {units.get(name)!r}")
        values = daily.get(name)
        if not isinstance(values, list) or len(values) != len(days):
            raise FetchError(f"{name}: missing or not one value per day")
        columns[column] = values
    df = pd.DataFrame(columns, index=pd.DatetimeIndex(pd.to_datetime(days), name="date"), dtype="float64")
    # ERA5 lags real time by about six days: the archive answers null for days it has not got yet.
    return df.dropna(how="all")


def download(
    city: City,
    start: date,
    end: date,
    *,
    get: GetJSON = get_json,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """All days from start to end, requested in CHUNK_YEARS-long pieces."""
    frames: list[pd.DataFrame] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(date(chunk_start.year + CHUNK_YEARS, 1, 1) - timedelta(days=1), end)
        if frames:
            sleep(PAUSE_SECONDS)
        frames.append(parse_response(get(build_url(city, chunk_start, chunk_end))))
        chunk_start = chunk_end + timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="date"), dtype="float64")
    return pd.concat(frames)


def csv_path(data_dir: Path, city: City) -> Path:
    return data_dir / f"{city.id}.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="date", date_format="%Y-%m-%d", lineterminator="\n")


def merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Old series with new values on top: a re-downloaded day replaces the stored one."""
    kept = old[~old.index.isin(new.index)]
    return pd.concat([kept, new]).sort_index()[COLUMNS]


def update_city(
    city: City,
    data_dir: Path,
    *,
    today: date,
    full: bool = False,
    get: GetJSON = get_json,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Bring data/raw/<city>.csv up to today. Returns whether the file changed."""
    path = csv_path(data_dir, city)
    before = path.read_bytes() if path.exists() else b""
    old = read_csv(path) if before and not full else None

    if old is None or old.empty:
        start = FIRST_DATE
    else:
        start = max(FIRST_DATE, old.index[-1].date() - timedelta(days=REFRESH_DAYS))
    new = download(city, start, today, get=get, sleep=sleep)
    if new.empty:
        raise FetchError(f"{city.id}: no data from {start} to {today}")

    write_csv(new[COLUMNS] if old is None else merge(old, new), path)
    return path.read_bytes() != before
