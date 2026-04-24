#!/usr/bin/env python3
"""
Скрипт для обучения моделей классификации новостей.
Поддерживает различные модели и конфигурации с GPU-ускорением.
"""

import argparse
import sys
import os
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data.preprocessing import TextPreprocessor
from src.models.logistic_regression import TfidfLogisticClassifier
from src.models.bert_classifier import BertClassifier
from src.models.cnn_word2vec import Word2VecCNNClassifier
from src.models.ensemble import VotingEnsemble, StackingEnsemble
from src.utils.config import load_config
from src.utils.logger import ExperimentLogger
from src.utils.artifact_manager import ArtifactManager


def print_gpu_info():
    """Вывод информации о доступных GPU."""
    print("\n" + "="*50)
    print("🖥️  СИСТЕМНАЯ ИНФОРМАЦИЯ")
    print("="*50)

    print(f"Python: {sys.version.split()[0]}")

    if torch.cuda.is_available():
        print(f"✅ PyTorch CUDA доступен!")
        print(f"   Версия CUDA: {torch.version.cuda}")
        print(f"   Версия cuDNN: {torch.backends.cudnn.version()}")
        print(f"   Доступно GPU: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name}")
            print(f"      Память: {memory_gb:.2f} GB")
            print(f"      Compute Capability: {props.major}.{props.minor}")

        # Рекомендации по batch size
        if torch.cuda.get_device_properties(0).total_memory > 12 * (1024**3):
            print("\n💡 Рекомендуемый batch size для BERT: 64-128")
        elif torch.cuda.get_device_properties(0).total_memory > 8 * (1024**3):
            print("\n💡 Рекомендуемый batch size для BERT: 32-64")
        else:
            print("\n💡 Рекомендуемый batch size для BERT: 16-32")
            print("   Используйте gradient_accumulation_steps для большего эффективного batch")
    else:
        print("⚠️  CUDA недоступна. Обучение будет на CPU.")
        print("   Для GPU установите PyTorch с CUDA поддержкой:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

    print("="*50 + "\n")


def set_seed(seed: int):
    """Установка seed для воспроизводимости."""
    import random

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Детерминированное поведение
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_data(config, use_gpu_preprocessing: bool = True):
    """Загрузка и разделение данных с GPU-оптимизацией."""
    import time

    start_time = time.time()

    # Загрузка
    train_path = Path(config.data.raw_path) / config.data.train_file

    if not train_path.exists():
        print(f"❌ Ошибка: Файл данных не найден: {train_path}")
        print("Поместите train.csv в data/raw/ с колонками: class, title, description")
        sys.exit(1)

    print(f"📁 Загрузка данных из {train_path}...")
    df = pd.read_csv(train_path)
    print(f"   Загружено {len(df)} записей")

    # Объединение title и description если есть
    if 'title' in df.columns and 'description' in df.columns:
        df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    elif 'title' in df.columns and 'text' not in df.columns:
        df['text'] = df['title']

    if 'class_name' in df.columns and 'class' not in df.columns:
        df['class'] = df['class_name']
    elif 'class_index' in df.columns and 'class' not in df.columns:
        df['class'] = df['class_index']

    # Предобработка с GPU-ускорением
    print("🔤 Предобработка текстов...")
    preprocessor = TextPreprocessor(
        strategy=config.preprocessing.strategy,
        lowercase=config.preprocessing.lowercase,
        remove_stopwords=config.preprocessing.remove_stopwords,
        lemmatize=config.preprocessing.lemmatize,
        use_gpu=use_gpu_preprocessing,
        batch_size=1000
    )

    # Batch preprocessing для скорости
    df['text_processed'] = preprocessor.preprocess_batch(
        df['text'].tolist(),
        n_jobs=-1,
        use_gpu=use_gpu_preprocessing
    )

    # Разделение
    X = df['text_processed']
    y = df['class']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=config.split.val_ratio + config.split.test_ratio,
        stratify=y if config.split.stratify else None,
        random_state=config.split.random_state
    )

    val_size = config.split.val_ratio / (config.split.val_ratio + config.split.test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_size,
        stratify=y_temp if config.split.stratify else None,
        random_state=config.split.random_state
    )

    elapsed = time.time() - start_time
    print(f"✅ Данные подготовлены за {elapsed:.2f} сек")
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(
    model_name: str,
    config,
    X_train,
    y_train,
    X_val,
    y_val,
    logger,
    artifact_manager,
    use_gpu: bool = True,
    bert_batch_size: int | None = None,
    bert_epochs: int | None = None,
    args=None,
):
    """Обучение конкретной модели с GPU-оптимизацией."""
    import time

    start_time = time.time()

    if model_name == "tfidf_logreg":
        model = TfidfLogisticClassifier(use_gpu=use_gpu)
    elif model_name == "bert":
        batch_size = bert_batch_size if bert_batch_size is not None else (64 if use_gpu else 16)
        epochs = bert_epochs if bert_epochs is not None else 3

        # BERT конфигурация с GPU
        model_config = {
            'model_name': 'distilbert-base-uncased',
            'num_labels': len(np.unique(y_train)),
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': 2e-5,
            'device': 'auto',  # Автоматический выбор GPU
            'mixed_precision': use_gpu,  # Mixed precision на GPU
            'gradient_accumulation_steps': 2 if use_gpu else 1,
            'num_workers': 4 if use_gpu else 0,
            'pin_memory': use_gpu
        }
        model = BertClassifier(model_config)
    elif model_name == "cnn":
        cnn_config = {
            'epochs': args.epochs if args and args.epochs is not None else 5,
            'batch_size': args.batch_size if args and args.batch_size is not None else (256 if use_gpu else 64),
            'embedding_dim': 200,
            'max_length': 80,
            'learning_rate': 1e-3,
            'num_workers': 2 if use_gpu else 0,
            'pin_memory': use_gpu,
            'eval_batch_size': 512,
        }
        model = Word2VecCNNClassifier(cnn_config, use_gpu=use_gpu)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    # Логирование параметров
    logger.log_param("model", model_name)
    logger.log_param("use_gpu", use_gpu)
    logger.log_params(model.config if hasattr(model, 'config') else {})

    # Обучение
    print(f"🚀 Обучение {model_name}...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val))

    # Сохранение модели
    model_path = Path(config.models.serialized_path) / f"{model_name}_{logger.run_id}"
    model.save(model_path)
    logger.log_model(str(model_path), model_name)
    artifact_manager.save_model_info(str(model_path), model_name, model.get_metadata())

    elapsed = time.time() - start_time
    print(f"✅ Обучение завершено за {elapsed:.2f} сек ({elapsed/60:.2f} мин)")

    return model


def evaluate_model(model, X_test, y_test, model_name, class_names, logger, artifact_manager):
    """Оценка модели."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import time

    print(f"\n📊 Оценка модели {model_name}...")
    start_time = time.time()

    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_macro": float(f1)
    }

    elapsed = time.time() - start_time

    # Логирование
    logger.log_metrics(metrics)
    artifact_manager.save_metrics(metrics, model_name)
    artifact_manager.save_predictions(
        y_test.values, y_pred, y_proba, model_name
    )
    artifact_manager.save_confusion_matrix(
        y_test.values, y_pred, class_names, model_name
    )
    artifact_manager.save_classification_report(
        y_test.values, y_pred, class_names, model_name
    )

    # Сохранение кривых обучения (если есть)
    if hasattr(model, 'training_history') and model.training_history:
        artifact_manager.save_training_curves(model.training_history, model_name)

    print(f"\n📈 Результаты {model_name}:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-macro:  {f1:.4f}")
    print(f"   Время инференса: {elapsed:.2f} сек")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train news classification models with GPU support")
    parser.add_argument("--model", type=str, required=True,
                       choices=["tfidf_logreg", "bert", "cnn", "voting", "stacking"],
                       help="Модель для обучения")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                       help="Путь к конфигу")
    parser.add_argument("--experiment-name", type=str, default="news_classification",
                       help="Название эксперимента")
    parser.add_argument("--use-mlflow", action="store_true",
                       help="Использовать MLflow")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Отключить GPU (принудительно CPU)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Размер batch для BERT (переопределяет конфиг)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Количество эпох для BERT")

    args = parser.parse_args()

    # Вывод информации о GPU
    print_gpu_info()

    # Определение использования GPU
    use_gpu = not args.no_gpu and torch.cuda.is_available()

    if use_gpu:
        print("✅ Используется GPU ускорение\n")
    else:
        print("⚠️  Используется CPU\n")

    # Загрузка конфигурации
    config = load_config(args.config)

    # Установка seed
    set_seed(config.project.seed)

    # Инициализация логгера
    with ExperimentLogger(
        experiment_name=args.experiment_name,
        use_mlflow=args.use_mlflow
    ) as logger:

        # Инициализация менеджера артефактов
        artifact_manager = ArtifactManager(logger.run_id)

        # Загрузка данных
        X_train, X_val, X_test, y_train, y_val, y_test = load_data(config, use_gpu_preprocessing=use_gpu)

        class_names = [str(i) for i in sorted(y_train.unique())]

        logger.log_param("num_samples_train", len(X_train))
        logger.log_param("num_samples_val", len(X_val))
        logger.log_param("num_samples_test", len(X_test))
        logger.log_param("num_classes", len(class_names))
        logger.log_param("gpu_available", torch.cuda.is_available())
        logger.log_param("gpu_used", use_gpu)

        # Обучение и оценка
        if args.model in ["voting", "stacking"]:
            # Ансамбли требуют предобученных моделей
            print("🔄 Обучение базовых моделей для ансамбля...")
            base_models = []
            for base_model_name in ["tfidf_logreg", "bert"]:
                base_model = train_model(
                    base_model_name, config, X_train, y_train,
                    X_val, y_val, logger, artifact_manager, use_gpu,
                    bert_batch_size=args.batch_size,
                    bert_epochs=args.epochs,
                    args=args,
                )
                base_models.append(base_model)

            if args.model == "voting":
                model = VotingEnsemble(base_models)
            else:
                model = StackingEnsemble(base_models)

            model.fit(X_train, y_train)
        else:
            model = train_model(
                args.model, config, X_train, y_train,
                X_val, y_val, logger, artifact_manager, use_gpu,
                bert_batch_size=args.batch_size,
                bert_epochs=args.epochs,
                args=args,
            )

        # Оценка
        metrics = evaluate_model(
            model, X_test, y_test, args.model,
            class_names, logger, artifact_manager
        )

        # Сохранение сводки
        artifact_manager.save_summary()

        print(f"\n" + "="*50)
        print(f"✅ Эксперимент завершен: {logger.run_id}")
        print(f"📁 Артефакты сохранены в: {artifact_manager.experiment_path}")
        print(f"💾 Модель сохранена в: {config.models.serialized_path}")
        print("="*50)


if __name__ == "__main__":
    main()
