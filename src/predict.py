"""Пакетное предсказание для новой порции данных."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from src.featurize import build
from src.utils import ensure_parent, get_logger, load_params

log = get_logger("predict")
MODEL_FILE = Path("models/model.joblib")

def load_local() -> tuple:
    bundle = joblib.load(MODEL_FILE)
    return bundle["model"], bundle["features"]

def load_from_registry(model_name: str, alias: str) -> tuple:
    import mlflow
    model = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
    return model, list(model.feature_names_in_)

def prepare_features(frame: pd.DataFrame, params: dict, features: list[str]) -> pd.DataFrame:
    target = params["data"]["target"]
    built = build(frame, target=target, drop=list(params["data"]["drop"]), cyclic_hour=bool(params["featurize"]["cyclic_hour"]))
    return built.reindex(columns=features)

def main() -> None:
    params = load_params()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/predictions.csv"))
    parser.add_argument("--from-registry", default=None, help="алиас модели в реестре")
    args = parser.parse_args()

    if args.from_registry:
        model, features = load_from_registry(params["mlflow"]["model_name"], args.from_registry)
        source = f"реестр, алиас {args.from_registry}"
    else:
        model, features = load_local()
        source = str(MODEL_FILE)

    frame = pd.read_csv(args.input, parse_dates=["dteday"])
    x_new = prepare_features(frame, params, features)
    predictions = model.predict(x_new)

    out = frame[["dteday", "hr"]].copy()
    out["prediction"] = predictions.round(1)
    out.to_csv(ensure_parent(args.output), index=False)
    log.info("модель: %s; предсказано %d строк -> %s", source, len(out), args.output)

if __name__ == "__main__":
    main()
