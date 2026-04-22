"""
Logistic Regression с TF-IDF признаками и GPU-ускорением (cuML/RAPIDS).
Поддерживает CPU fallback для совместимости.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from .base import BaseClassifier

# Попытка импорта cuML для GPU-ускорения
try:
    from cuml.linear_model import LogisticRegression as cuMLLogisticRegression
    from cuml.feature_extraction.text import TfidfVectorizer as cuTfidfVectorizer
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TfidfLogisticClassifier(BaseClassifier):
    """
    Классификатор на основе TF-IDF и Logistic Regression с GPU-ускорением.

    При наличии CUDA и cuML (RAPIDS) использует GPU-ускорение,
    иначе fallback на sklearn (CPU).

    Args:
        config: Конфигурация модели
        use_gpu: Принудительное использование GPU (если доступно)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, use_gpu: bool = True):
        super().__init__(config)

        self.use_gpu = use_gpu and CUML_AVAILABLE

        self.config = config or {
            'tfidf': {
                'max_features': 10000,
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.95
            },
            'classifier': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            }
        }

        if self.use_gpu:
            print("✅ Используется GPU-ускорение (cuML/RAPIDS)")
            self.vectorizer = cuTfidfVectorizer(**self.config['tfidf'])
            self.classifier = cuMLLogisticRegression(**self.config['classifier'])
        else:
            if CUML_AVAILABLE:
                print("⚠️  GPU недоступен, используется CPU (sklearn)")
            else:
                print("⚠️  cuML не установлен, используется CPU (sklearn)")
                print("   Для GPU установите: pip install cuml-cu11")

            self.vectorizer = TfidfVectorizer(**self.config['tfidf'])
            self.classifier = LogisticRegression(**self.config['classifier'])

        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('clf', self.classifier)
        ])

        self.model = self.pipeline

    def fit(self, X, y, validation_data=None):
        """Обучение модели с оптимизацией для GPU."""
        import time

        start_time = time.time()
        print("🚀 Обучение TF-IDF + Logistic Regression...")

        self.model.fit(X, y)
        self.is_fitted = True

        # Сохранение информации о классах
        self.metadata['classes'] = list(self.model.classes_)
        self.metadata['num_features'] = len(self.vectorizer.get_feature_names_out())
        self.metadata['use_gpu'] = self.use_gpu

        elapsed = time.time() - start_time
        print(f"✅ Обучение завершено за {elapsed:.2f} сек")

        # Валидация
        if validation_data is not None:
            self._validate(validation_data)

    def _validate(self, validation_data):
        """Быстрая валидация."""
        from sklearn.metrics import accuracy_score

        X_val, y_val = validation_data
        predictions = self.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        print(f"📊 Validation Accuracy: {accuracy:.4f}")
        return accuracy

    def predict(self, X) -> np.ndarray:
        """Предсказание классов."""
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Предсказание вероятностей."""
        return self.model.predict_proba(X)

    def get_feature_importance(self, class_idx: int, top_n: int = 20):
        """
        Получение важности признаков для класса.

        Returns:
            Список (feature, weight) отсортированный по весу.
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена")

        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.classifier.coef_[class_idx]

        top_indices = np.argsort(np.abs(coefficients))[-top_n:][::-1]
        return [(feature_names[i], coefficients[i]) for i in top_indices]

    def to_gpu(self):
        """Перенос обученной модели на GPU (если доступно)."""
        if not CUML_AVAILABLE:
            print("⚠️  cuML не установлен. Невозможно перенести на GPU.")
            return

        if self.use_gpu:
            print("✅ Модель уже на GPU")
            return

        print("🔄 Перенос модели на GPU...")
        # Пересоздание моделей на GPU
        self.classifier = cuMLLogisticRegression(**self.config['classifier'])

        # Копирование весов (при необходимости переобучить)
        print("⚠️  Для GPU требуется переобучение модели")
        self.use_gpu = True

    def save(self, path: str, format: str = "joblib"):
        """Сохранение модели с информацией о GPU."""
        super().save(path, format)
        # Дополнительно сохраняем информацию о GPU
        import json
        from pathlib import Path
        gpu_info_path = Path(path).parent / f"{Path(path).stem}_gpu_info.json"
        with open(gpu_info_path, 'w') as f:
            json.dump({'use_gpu': self.use_gpu, 'cuml_available': CUML_AVAILABLE}, f)
