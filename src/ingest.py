import argparse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.utils import ensure_parent, get_logger

log = get_logger("ingest")
DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def download_raw(target_path: Path):
    if not target_path.exists():
        ensure_parent(target_path)
        zip_path = target_path.parent / "bike.zip"
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extract("hour.csv", target_path.parent)
        zip_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-year", type=int, default=2011)
    parser.add_argument("--until-year", type=int, default=2012)
    parser.add_argument("--out", type=Path, default=Path("data/raw/hour.csv"))
    args = parser.parse_args()

    download_raw(Path("data/raw/hour.csv"))
    df = pd.read_csv("data/raw/hour.csv", parse_dates=["dteday"])
    df_filtered = df[
        (df["dteday"].dt.year >= args.from_year)
        & (df["dteday"].dt.year <= args.until_year)
    ]
    df_filtered.to_csv(ensure_parent(args.out), index=False)
    log.info(
        "выгрузка %d—%d: %d строк -> %s",
        args.from_year,
        args.until_year,
        len(df_filtered),
        args.out,
    )


if __name__ == "__main__":
    main()
