from __future__ import annotations

import os
from pathlib import Path
from typing import Any

RUN_ID_FILE = Path("reports/run_id.txt")
MODEL_URI_FILE = Path("reports/model_uri.txt")


def enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def start_run(experiment: str, params: dict[str, Any]):
    if not enabled():
        return None
    import mlflow

    mlflow.set_experiment(experiment)
    run = mlflow.start_run()
    mlflow.log_params(params)
    RUN_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUN_ID_FILE.write_text(run.info.run_id, encoding="utf-8")
    return run


def log_metrics(metrics: dict[str, float]) -> None:
    if not enabled() or not RUN_ID_FILE.exists():
        return
    import mlflow

    run_id = RUN_ID_FILE.read_text(encoding="utf-8").strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)


def log_model(model, features: list[str], example) -> None:
    if not enabled():
        return
    import cloudpickle
    import mlflow
    import pandas
    import sklearn
    from mlflow.models import infer_signature

    signature = infer_signature(example, model.predict(example))
    requirements = [
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pandas.__version__}",
        f"cloudpickle=={cloudpickle.__version__}",
    ]
    info = mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=example.head(3),
        serialization_format="cloudpickle",
        pip_requirements=requirements,
    )
    MODEL_URI_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_URI_FILE.write_text(info.model_uri, encoding="utf-8")
