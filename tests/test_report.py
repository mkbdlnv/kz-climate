import json
from datetime import UTC, date, datetime, timedelta

import pytest

from kzclimate import fetch, report
from kzclimate.__main__ import main
from kzclimate.cities import CITIES

from .conftest import synthetic_daily

ALMATY = CITIES[0]


@pytest.fixture(scope="module")
def city(daily):
    return report.city_report(ALMATY, daily)


def test_city_covers_complete_years_only(city):
    assert city["years"][0]["year"] == 1941
    assert city["years"][-1]["year"] == 2025
    assert city["last_year"] == {"year": 2025, "rank": city["last_year"]["rank"], "of": 85}
    assert city["data_from"] == "1940-01-02"
    assert city["data_through"] == "2026-10-03"


def test_anomalies_are_relative_to_1961_1990(city):
    baseline = [y["anomaly"] for y in city["years"] if 1961 <= y["year"] <= 1990]
    assert sum(baseline) / len(baseline) == pytest.approx(0, abs=0.01)


def test_finds_the_built_in_warming(city):
    t = city["trends"]["mean"]
    assert t["per_decade"] == pytest.approx(0.3, abs=0.05)  # synthetic series warms 0.03 °C a year
    assert t["significant"] is True
    assert t["p_value"] < 1e-6


def test_warmest_years_are_sorted_and_recent(city):
    anomalies = [w["anomaly"] for w in city["warmest"]]
    assert anomalies == sorted(anomalies, reverse=True)
    assert all(w["year"] >= 1990 for w in city["warmest"])


def test_has_twelve_normals_and_recent_months(city):
    assert len(city["normals"]) == 12
    assert len(city["recent_months"]) == 12


def test_whole_file_is_strict_json(daily):
    now = datetime(2026, 10, 9, 3, 0, tzinfo=UTC)
    result = report.build({ALMATY: daily}, version="v-test", commit="abc123", now=now)
    text = json.dumps(result, ensure_ascii=False, allow_nan=False)
    back = json.loads(text)
    assert back["schema"] == report.SCHEMA
    assert back["version"] == "v-test"
    assert back["generated_at"] == "2026-10-09T03:00:00+00:00"
    assert back["code"] == {"repository": report.REPOSITORY, "commit": "abc123"}
    assert [c["id"] for c in back["cities"]] == ["almaty"]


def test_command_line_builds_the_file(tmp_path, capsys):
    end = datetime.now(UTC).date() - timedelta(days=6)
    for i, c in enumerate(CITIES):
        fetch.write_csv(synthetic_daily(end=end.isoformat(), seed=i), fetch.csv_path(tmp_path, c))
    out = tmp_path / "dist" / "climate.json"

    assert main(["--data-dir", str(tmp_path), "build", "--out", str(out), "--version", "v1"]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert [c["id"] for c in result["cities"]] == ["almaty", "astana"]
    assert "°C/decade" in capsys.readouterr().out


def test_command_line_refuses_bad_data(tmp_path, capsys):
    df = synthetic_daily(end=(date.today() - timedelta(days=6)).isoformat())
    df.iloc[100, 1] = 99.0
    fetch.write_csv(df, fetch.csv_path(tmp_path, ALMATY))
    out = tmp_path / "climate.json"

    assert main(["--data-dir", str(tmp_path), "--city", "almaty", "build", "--out", str(out)]) == 1
    assert not out.exists()
    assert "tmax outside" in capsys.readouterr().err
