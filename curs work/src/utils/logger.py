"""
Система логирования экспериментов с поддержкой MLflow.
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# MLflow опционален
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentLogger:
    """
    Логгер для отслеживания ML-экспериментов.
    Поддерживает локальное логирование и MLflow.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "./logs/experiments",
        use_mlflow: bool = False,
        mlflow_tracking_uri: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.experiment_path = self.log_dir / experiment_name
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE

        # Уникальный ID эксперимента
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.experiment_path / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Настройка файлового логгера
        self._setup_logger()

        # Инициализация MLflow
        if self.use_mlflow:
            self._setup_mlflow(mlflow_tracking_uri)

        # Метаданные эксперимента
        self.metadata = {
            "experiment_name": experiment_name,
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
            "params": {},
            "metrics": {},
            "artifacts": []
        }

    def _setup_logger(self):
        """Настройка Python logger."""
        self.logger = logging.getLogger(f"{self.experiment_name}_{self.run_id}")
        self.logger.setLevel(logging.INFO)

        # Очистка старых обработчиков
        self.logger.handlers = []

        # File handler
        fh = logging.FileHandler(self.run_dir / "experiment.log")
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _setup_mlflow(self, tracking_uri: Optional[str]):
        """Настройка MLflow."""
        if not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow not installed, skipping MLflow setup")
            return

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_id)
        self.logger.info(f"MLflow tracking started: {tracking_uri}")

    def log_param(self, key: str, value: Any):
        """Логирование параметра."""
        self.metadata["params"][key] = value
        self.logger.info(f"Param: {key} = {value}")

        if self.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]):
        """Логирование нескольких параметров."""
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Логирование метрики."""
        self.metadata["metrics"][key] = value
        self.logger.info(f"Metric: {key} = {value}")

        if self.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Логирование нескольких метрик."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Логирование артефакта."""
        self.metadata["artifacts"].append({
            "local_path": local_path,
            "artifact_path": artifact_path
        })
        self.logger.info(f"Artifact logged: {local_path}")

        if self.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_artifact(local_path, artifact_path)

    def log_dict(self, data: Dict, filename: str):
        """Сохранение словаря как JSON."""
        filepath = self.run_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        self.log_artifact(str(filepath))

    def log_model(self, model_path: str, model_name: str):
        """Логирование модели."""
        self.logger.info(f"Model logged: {model_name}")

        if self.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_artifact(model_path, "models")

    def finish(self):
        """Завершение эксперимента."""
        self.metadata["end_time"] = datetime.now().isoformat()

        # Сохранение метаданных
        metadata_path = self.run_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

        self.logger.info(f"Experiment finished. Metadata saved to {metadata_path}")

        if self.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.end_run()

    def __enter__(self):
        """Контекстный менеджер."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Завершение при выходе из контекста."""
        self.finish()
