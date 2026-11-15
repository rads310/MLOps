"""Реестр моделей MLflow."""
from __future__ import annotations
import argparse
from pathlib import Path
from src.utils import get_logger, load_params

log = get_logger("registry")
RUN_ID_FILE = Path("reports/run_id.txt")
MODEL_URI_FILE = Path("reports/model_uri.txt")

def _client():
    from mlflow import MlflowClient
    return MlflowClient()

def _metric(client, version, name: str) -> float | None:
    run = client.get_run(version.run_id)
    return run.data.metrics.get(name)

def register(model_name: str) -> str:
    import mlflow
    if MODEL_URI_FILE.exists():
        model_uri = MODEL_URI_FILE.read_text(encoding="utf-8").strip()
    else:
        model_uri = f"runs:/{RUN_ID_FILE.read_text(encoding='utf-8').strip()}/model"
    version = mlflow.register_model(model_uri, model_name)
    _client().set_registered_model_alias(model_name, "challenger", version.version)
    log.info("зарегистрирована версия %s, алиас challenger", version.version)
    return version.version

def promote(model_name: str, metric: str, greater_is_better: bool = True) -> bool:
    client = _client()
    challenger = client.get_model_version_by_alias(model_name, "challenger")
    new_value = _metric(client, challenger, metric)

    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except Exception:
        champion = None

    if champion is None:
        client.set_registered_model_alias(model_name, "champion", challenger.version)
        log.info("champion не был задан: версия %s становится champion", challenger.version)
        return True

    old_value = _metric(client, champion, metric)
    better = new_value > old_value if greater_is_better else new_value < old_value
    log.info(
        "%s: challenger v%s = %.4f, champion v%s = %.4f -> %s",
        metric, challenger.version, new_value, champion.version, old_value,
        "продвигаем" if better else "оставляем прежнего",
    )
    if better:
        client.set_registered_model_alias(model_name, "champion", challenger.version)
    return better

def main() -> None:
    params = load_params()["mlflow"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["register", "promote"])
    parser.add_argument("--model-name", default=params["model_name"])
    parser.add_argument("--metric", default=params["primary_metric"])
    args = parser.parse_args()

    if args.action == "register":
        register(args.model_name)
    else:
        promote(args.model_name, args.metric)

if __name__ == "__main__":
    main()
