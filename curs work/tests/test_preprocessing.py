"""
Тесты для модуля предобработки.
"""

import pytest
from src.data.preprocessing import TextPreprocessor


def test_text_preprocessor_clean():
    """Тест базовой очистки."""
    processor = TextPreprocessor(strategy='classic')

    text = "Check out https://example.com and email@test.com!!!"
    cleaned = processor.clean_text(text)

    assert "https://example.com" not in cleaned
    assert "email@test.com" not in cleaned


def test_text_preprocessor_bert_strategy():
    """Тест стратегии BERT (минимальная обработка)."""
    processor = TextPreprocessor(strategy='bert', lowercase=False)

    text = "Hello World!"
    result = processor.preprocess(text)

    assert "Hello" in result or "hello" in result


def test_text_preprocessor_batch():
    """Тест пакетной обработки."""
    processor = TextPreprocessor(strategy='classic', lowercase=True)

    texts = ["Hello World!", "Test TEXT"]
    results = processor.preprocess_batch(texts)

    assert len(results) == len(texts)
    assert all(isinstance(r, str) for r in results)
