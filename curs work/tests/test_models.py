"""
Тесты для моделей.
"""

import pytest
import numpy as np
from src.models.logistic_regression import TfidfLogisticClassifier


def test_tfidf_classifier_fit_predict():
    """Тест обучения и предсказания."""
    model = TfidfLogisticClassifier()

    X = np.array(["this is class one", "this is class two"] * 10)
    y = np.array([0, 1] * 10)

    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_tfidf_classifier_predict_proba():
    """Тест предсказания вероятностей."""
    model = TfidfLogisticClassifier()

    X = np.array(["text one", "text two"] * 10)
    y = np.array([0, 1] * 10)
    model.fit(X, y)

    probas = model.predict_proba(X)

    assert probas.shape == (len(X), 2)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_model_save_load(tmp_path):
    """Тест сохранения и загрузки."""
    model = TfidfLogisticClassifier()

    X = np.array(["text one", "text two"] * 10)
    y = np.array([0, 1] * 10)
    model.fit(X, y)

    # Сохранение
    save_path = tmp_path / "test_model"
    model.save(save_path)

    # Загрузка
    loaded_model = TfidfLogisticClassifier.load(save_path)
    predictions = loaded_model.predict(X)

    assert len(predictions) == len(y)
