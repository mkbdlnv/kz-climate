"""Command line: python -m kzclimate fetch | check | build.

  fetch   bring data/raw/<city>.csv up to date from Open-Meteo
  check   validate data/raw without building anything
  build   validate and write dist/climate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from . import fetch, report, validate
from .cities import CITIES, by_id


def _today() -> date:
    return datetime.now(UTC).date()


def _cities(ids: list[str] | None):
    return [by_id(i) for i in ids] if ids else list(CITIES)


def _load(data_dir: Path, ids: list[str] | None, today: date):
    data = {}
    for city in _cities(ids):
        path = fetch.csv_path(data_dir, city)
        if not path.exists():
            raise validate.DataError(city.id, [f"{path} not found; run `python -m kzclimate fetch` first"])
        df = fetch.read_csv(path)
        validate.check(city.id, df, today=today)
        data[city] = df
    return data


def cmd_fetch(args: argparse.Namespace) -> int:
    for city in _cities(args.city):
        changed = fetch.update_city(city, args.data_dir, today=_today(), full=args.full)
        print(f"{city.id}: {'updated' if changed else 'no new data'}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    data = _load(args.data_dir, args.city, _today())
    for city, df in data.items():
        print(f"{city.id}: ok, {len(df)} days, {df.index[0].date()} – {df.index[-1].date()}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    data = _load(args.data_dir, args.city, _today())
    result = report.build(data, version=args.version, commit=args.commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: a NaN that slipped through fails here, not in the dashboard.
    args.out.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for city in result["cities"]:
        t = city["trends"]["mean"]
        print(
            f"{city['id']}: {city['years'][0]['year']}–{city['last_year']['year']}, "
            f"{t['per_decade']:+.2f} °C/decade (p={t['p_value']:.2g}), "
            f"{city['last_year']['year']} ranks {city['last_year']['rank']} of {city['last_year']['of']}"
        )
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kzclimate", description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--city", action="append", choices=[c.id for c in CITIES], help="only this city (repeatable)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download new days into data/raw")
    p.add_argument("--full", action="store_true", help="download everything since 1940 again")
    p.set_defaults(run=cmd_fetch)

    p = sub.add_parser("check", help="validate data/raw")
    p.set_defaults(run=cmd_check)

    p = sub.add_parser("build", help="validate data/raw and write climate.json")
    p.add_argument("--out", type=Path, default=Path("dist/climate.json"))
    p.add_argument("--version", default="dev", help="release tag written into the file")
    p.add_argument("--commit", default=None, help="git commit the file was built from")
    p.set_defaults(run=cmd_build)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (validate.DataError, fetch.FetchError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
