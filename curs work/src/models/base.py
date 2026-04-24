"""
Базовый класс для всех моделей классификации.
Обеспечивает единый интерфейс для обучения, предсказания и сериализации.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
import pickle
import joblib
from pathlib import Path
import json


class BaseClassifier(ABC):
    """
    Абстрактный базовый класс для классификаторов.

    Все наследники должны реализовать:
    - fit: обучение модели
    - predict: предсказание классов
    - predict_proba: предсказание вероятностей (опционально)
    - save: сохранение модели
    - load: загрузка модели
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.metadata = {
            "model_type": self.__class__.__name__,
            "config": self.config
        }

    @abstractmethod
    def fit(self, X, y, validation_data=None):
        """
        Обучение модели.

        Args:
            X: Признаки
            y: Целевые метки
            validation_data: Кортеж (X_val, y_val) для валидации
        """
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Предсказание классов."""
        pass

    def predict_proba(self, X) -> Optional[np.ndarray]:
        """
        Предсказание вероятностей классов.

        Returns:
            Массив вероятностей или None, если не поддерживается.
        """
        return None

    def get_metadata(self) -> Dict[str, Any]:
        """Получение метаданных модели."""
        return self.metadata

    def save(self, path: Union[str, Path], format: str = "joblib"):
        """
        Сохранение модели.

        Args:
            path: Путь для сохранения
            format: Формат сериализации ("pickle", "joblib", "json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Сохранение метаданных
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

        # Сохранение модели
        if format == "pickle":
            with open(path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(self.model, f)
        elif format == "joblib":
            joblib.dump(self.model, path.with_suffix('.joblib'))
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load(cls, path: Union[str, Path], format: str = "joblib"):
        """
        Загрузка модели.

        Args:
            path: Путь к файлу модели
            format: Формат сериализации

        Returns:
            Экземпляр классификатора
        """
        path = Path(path)
        instance = cls()

        if format == "pickle":
            with open(path.with_suffix('.pkl'), 'rb') as f:
                instance.model = pickle.load(f)
        elif format == "joblib":
            instance.model = joblib.load(path.with_suffix('.joblib'))

        # Загрузка метаданных
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                instance.metadata = json.load(f)

        instance.is_fitted = True
        return instance
