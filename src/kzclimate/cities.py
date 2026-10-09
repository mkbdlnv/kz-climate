"""Places the analysis covers. A new city is one more line in CITIES."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class City:
    id: str  # file name in data/raw and key in climate.json
    name: str
    lat: float
    lon: float


CITIES: tuple[City, ...] = (
    City("almaty", "Алматы", 43.2567, 76.9286),
    City("astana", "Астана", 51.1694, 71.4491),
)

# Days are cut at local midnight. Both cities share this zone (UTC+5 since March 2024).
TIMEZONE = "Asia/Almaty"

# ERA5 starts on 1940-01-01 at 00:00 UTC, which is 05:00 or 06:00 local time, so the first
# local day is incomplete: Open-Meteo returns a null mean and partial min/max for it.
# The series starts with the first full local day; yearly statistics therefore start in 1941.
FIRST_DATE = date(1940, 1, 2)


def by_id(city_id: str) -> City:
    for city in CITIES:
        if city.id == city_id:
            return city
    raise KeyError(city_id)
