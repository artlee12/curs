"""
Ансамблевые методы: Voting и Stacking.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import joblib
from .base import BaseClassifier


class VotingEnsemble(BaseClassifier):
    """
    Hard Voting Ensemble.
    """

    def __init__(self, models: List[BaseClassifier], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.classes_ = None

    def fit(self, X, y, validation_data=None):
        """Обучение всех базовых моделей."""
        for model in self.models:
            model.fit(X, y, validation_data)

        # Получение классов из первой модели
        self.classes_ = getattr(self.models[0], 'classes_', np.unique(y))
        self.is_fitted = True

    def predict(self, X) -> np.ndarray:
        """Hard voting."""
        predictions = np.array([model.predict(X) for model in self.models])

        # Взвешенное голосование
        final_predictions = []
        for i in range(len(X)):
            votes = {}
            for pred, weight in zip(predictions[:, i], self.weights):
                votes[pred] = votes.get(pred, 0) + weight

            final_predictions.append(max(votes, key=votes.get))

        return np.array(final_predictions)

    def save(self, path, format: str = "joblib"):
        """Сохранение всех моделей ансамбля."""
        import os
        os.makedirs(path, exist_ok=True)

        # Сохранение каждой модели
        for i, model in enumerate(self.models):
            model_path = f"{path}/model_{i}"
            model.save(model_path, format)

        # Сохранение конфигурации ансамбля
        config = {
            'num_models': len(self.models),
            'weights': self.weights,
            'classes': self.classes_.tolist() if hasattr(self.classes_, 'tolist') else list(self.classes_)
        }
        joblib.dump(config, f"{path}/ensemble_config.joblib")


class StackingEnsemble(BaseClassifier):
    """
    Stacking с мета-классификатором.
    """

    def __init__(
        self,
        base_models: List[BaseClassifier],
        meta_model: Optional[Any] = None,
        use_probas: bool = True,
        cv_folds: int = 5
    ):
        super().__init__()
        self.base_models = base_models

        from sklearn.linear_model import LogisticRegression
        self.meta_model = meta_model or LogisticRegression(max_iter=1000)
        self.use_probas = use_probas
        self.cv_folds = cv_folds

    def fit(self, X, y, validation_data=None):
        """Обучение стекинга с out-of-fold предсказаниями."""
        from sklearn.model_selection import StratifiedKFold

        # Out-of-fold предсказания для обучения мета-модели
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        meta_features = []
        meta_labels = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]

            fold_meta_features = []

            for model in self.base_models:
                # Обучение на фолде
                model.fit(X_train, y_train)

                # Предсказание на валидационном фолде
                if self.use_probas:
                    proba = model.predict_proba(X_val)
                    fold_meta_features.append(proba)
                else:
                    preds = model.predict(X_val)
                    fold_meta_features.append(preds.reshape(-1, 1))

            # Объединение предсказаний всех моделей
            meta_features.append(np.hstack(fold_meta_features))
            meta_labels.append(y.iloc[val_idx])

        # Обучение мета-модели
        meta_X = np.vstack(meta_features)
        meta_y = np.concatenate(meta_labels)
        self.meta_model.fit(meta_X, meta_y)

        # Переобучение базовых моделей на всех данных
        for model in self.base_models:
            model.fit(X, y)

        self.is_fitted = True

    def predict(self, X) -> np.ndarray:
        """Предсказание через мета-модель."""
        meta_features = []

        for model in self.base_models:
            if self.use_probas:
                proba = model.predict_proba(X)
                meta_features.append(proba)
            else:
                preds = model.predict(X)
                meta_features.append(preds.reshape(-1, 1))

        meta_X = np.hstack(meta_features)
        return self.meta_model.predict(meta_X)

    def predict_proba(self, X) -> Optional[np.ndarray]:
        """Предсказание вероятностей через мета-модель."""
        meta_features = []

        for model in self.base_models:
            if self.use_probas:
                proba = model.predict_proba(X)
                meta_features.append(proba)
            else:
                preds = model.predict(X)
                meta_features.append(preds.reshape(-1, 1))

        meta_X = np.hstack(meta_features)

        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_X)
        return None
