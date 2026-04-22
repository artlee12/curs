#!/usr/bin/env python3
"""
Скрипт для инференса обученных моделей.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.logistic_regression import TfidfLogisticClassifier
from src.models.bert_classifier import BertClassifier


def main():
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to saved model")
    parser.add_argument("--model-type", type=str, required=True,
                       choices=["tfidf_logreg", "bert"],
                       help="Type of model")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input CSV with 'text' column")
    parser.add_argument("--output", type=str, default="predictions.csv",
                       help="Path to save predictions")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for prediction")

    args = parser.parse_args()

    # Загрузка модели
    print(f"Loading model from {args.model_path}...")
    if args.model_type == "tfidf_logreg":
        model = TfidfLogisticClassifier.load(args.model_path)
    elif args.model_type == "bert":
        model = BertClassifier.load(args.model_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Загрузка данных
    df = pd.read_csv(args.input)

    # Предобработка (если нужна)
    texts = df['text'].tolist()

    # Предсказание
    print("Making predictions...")
    predictions = model.predict(texts)

    # Вероятности (если поддерживается)
    probabilities = model.predict_proba(texts)

    # Сохранение результатов
    df['predicted_class'] = predictions

    if probabilities is not None:
        for i in range(probabilities.shape[1]):
            df[f'probability_class_{i}'] = probabilities[:, i]

    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"Total predictions: {len(predictions)}")


if __name__ == "__main__":
    main()
