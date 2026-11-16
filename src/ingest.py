from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import ensure_parent, get_logger, load_params

log = get_logger("ingest")


def main() -> None:
    params = load_params()
    ingest = params["ingest"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--until-year", type=int, default=ingest["until_year"])
    parser.add_argument("--from-year", type=int, default=None)
    parser.add_argument("--source", type=Path, default=Path(ingest["source"]))
    parser.add_argument("--out", type=Path, default=Path(params["data"]["raw"]))
    args = parser.parse_args()

    source = pd.read_csv(args.source, parse_dates=["dteday"])
    years = source["dteday"].dt.year
    sliced = source[years <= args.until_year]
    if args.from_year is not None:
        sliced = sliced[years >= args.from_year]
    sliced.to_csv(ensure_parent(args.out), index=False)
    log.info(
        "выгрузка до %d года: %d строк -> %s", args.until_year, len(sliced), args.out
    )


if __name__ == "__main__":
    main()
