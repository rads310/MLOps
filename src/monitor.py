"""Мониторинг: дрейф признаков и качество модели на новой порции данных."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.predict import MODEL_FILE, prepare_features
from src.utils import ensure_parent, get_logger, load_params

log = get_logger("monitor")
REFERENCE = Path("data/features/train.csv")
HTML_OUT = Path("reports/drift.html")
JSON_OUT = Path("reports/drift.json")


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, html_out: Path
) -> dict:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    columns = list(reference.columns)
    definition = DataDefinition(numerical_columns=columns)
    ref_ds = Dataset.from_pandas(reference, data_definition=definition)
    cur_ds = Dataset.from_pandas(current[columns], data_definition=definition)

    snapshot = Report([DataDriftPreset()]).run(cur_ds, ref_ds)
    snapshot.save_html(str(ensure_parent(html_out)))

    summary = {"drifted": 0, "share": 0.0, "columns": {}}
    for metric in snapshot.dict()["metrics"]:
        name, value = metric["metric_name"], metric["value"]
        if name.startswith("DriftedColumnsCount"):
            summary["drifted"] = int(value["count"])
            summary["share"] = round(float(value["share"]), 3)
        elif name.startswith("ValueDrift(column="):
            column = name.split("column=")[1].split(",")[0]
            summary["columns"][column] = round(float(value), 6)
    summary["n_features"] = len(columns)
    return summary


def main() -> None:
    params = load_params()
    monitor_params = params["monitor"]
    target = params["data"]["target"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--html", type=Path, default=HTML_OUT)
    parser.add_argument("--json", dest="json_out", type=Path, default=JSON_OUT)
    parser.add_argument("--fail-on-alert", action="store_true")
    args = parser.parse_args()

    reference = pd.read_csv(args.reference).drop(columns=[target], errors="ignore")
    raw_current = pd.read_csv(args.current, parse_dates=["dteday"])

    bundle = joblib.load(MODEL_FILE)
    model, features = bundle["model"], bundle["features"]
    current = prepare_features(raw_current, params, features)

    report = drift_report(reference[features], current, args.html)

    report["r2_stream"] = None
    report["mae_stream"] = None
    if target in raw_current.columns:
        y_true = raw_current[target]
        y_pred = model.predict(current)
        report["r2_stream"] = round(float(r2_score(y_true, y_pred)), 4)
        report["mae_stream"] = round(float(mean_absolute_error(y_true, y_pred)), 3)

    reasons = []
    if report["share"] > float(monitor_params["drift_share_threshold"]):
        reasons.append(
            f"дрейф: {report['share']:.0%} признаков при пороге {float(monitor_params['drift_share_threshold']):.0%}"
        )
    if report["r2_stream"] is not None and report["r2_stream"] < float(
        monitor_params["min_r2_on_stream"]
    ):
        reasons.append(
            f"качество: R2 = {report['r2_stream']} ниже порога {monitor_params['min_r2_on_stream']}"
        )

    report["status"] = "alert" if reasons else "ok"
    report["reasons"] = reasons
    report["n_rows"] = int(len(raw_current))
    ensure_parent(args.json_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log.info("дрейф: %s из %s признаков", report["drifted"], report["n_features"])
    log.info(
        "качество на потоке: R2 = %s, MAE = %s",
        report["r2_stream"],
        report["mae_stream"],
    )
    log.info("статус: %s %s", report["status"], "; ".join(reasons))

    if args.fail_on_alert and report["status"] == "alert":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
