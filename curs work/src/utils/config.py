"""
Утилиты для работы с конфигурациями.
"""

import yaml
import os
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class ProjectConfig:
    """Конфигурация проекта."""
    name: str = "news-classification"
    version: str = "1.0.0"
    seed: int = 42


@dataclass
class DataConfig:
    """Конфигурация данных."""
    raw_path: str = "./data/raw"
    processed_path: str = "./data/processed"
    embeddings_path: str = "./data/embeddings"
    train_file: str = "train.csv"
    test_file: str = "test.csv"


@dataclass
class PreprocessingConfig:
    """Конфигурация предобработки."""
    strategy: str = "classic"
    lowercase: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    max_length: int = 512


@dataclass
class SplitConfig:
    """Конфигурация разделения данных."""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify: bool = True
    random_state: int = 42


@dataclass
class ArtifactsConfig:
    """Конфигурация артефактов."""
    metrics_path: str = "./artifacts/metrics"
    predictions_path: str = "./artifacts/predictions"
    visualizations_path: str = "./artifacts/visualizations"
    confusion_matrices_path: str = "./artifacts/confusion_matrices"


@dataclass
class ModelsConfig:
    """Конфигурация моделей."""
    checkpoints_path: str = "./models/checkpoints"
    serialized_path: str = "./models/serialized"
    onnx_path: str = "./models/onnx"


@dataclass
class LoggingConfig:
    """Конфигурация логирования."""
    level: str = "INFO"
    log_dir: str = "./logs"
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "./logs/mlruns"


@dataclass
class Config:
    """Полная конфигурация."""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Загрузка конфигурации из YAML."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            project=ProjectConfig(**config_dict.get('project', {})),
            data=DataConfig(**config_dict.get('data', {})),
            preprocessing=PreprocessingConfig(**config_dict.get('preprocessing', {})),
            split=SplitConfig(**config_dict.get('split', {})),
            artifacts=ArtifactsConfig(**config_dict.get('artifacts', {})),
            models=ModelsConfig(**config_dict.get('models', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )

    def to_yaml(self, path: str):
        """Сохранение конфигурации в YAML."""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


def load_config(config_path: str = "configs/base_config.yaml") -> Config:
    """Загрузка конфигурации."""
    if os.path.exists(config_path):
        return Config.from_yaml(config_path)
    return Config()
