#!/usr/bin/env python3
"""
Скрипт для автоматической загрузки и подготовки датасета AG News.
Скачивает данные, сохраняет в CSV и выводит статистику.
"""

import os
import sys
import subprocess

# Пути относительно расположения скрипта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def install_requirements():
    """Устанавливает необходимые пакеты."""
    print("📦 Установка зависимостей...")
    packages = ["datasets", "pandas", "scikit-learn", "tqdm"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
    print("✅ Зависимости установлены\n")


def download_ag_news():
    """Скачивает датасет AG News через Hugging Face datasets."""
    print("📥 Загрузка датасета AG News...")

    from datasets import load_dataset
    import pandas as pd

    # Загружаем датасет
    dataset = load_dataset("ag_news", trust_remote_code=True)

    # Создаём директории
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Конвертируем в DataFrame
    train_df = pd.DataFrame({
        'class_index': dataset['train']['label'],
        'title': dataset['train']['text']  # AG News не разделяет title и description
    })

    test_df = pd.DataFrame({
        'class_index': dataset['test']['label'],
        'title': dataset['test']['text']
    })

    # Добавляем названия классов
    class_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    train_df['class_name'] = train_df['class_index'].map(class_names)
    test_df['class_name'] = test_df['class_index'].map(class_names)

    # Сохраняем
    train_path = os.path.join(RAW_DIR, "train.csv")
    test_path = os.path.join(RAW_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Датасет сохранён:")
    print(f"   - Train: {train_path} ({len(train_df):,} записей)")
    print(f"   - Test: {test_path} ({len(test_df):,} записей)\n")

    return train_df, test_df


def split_train_val(train_df, val_ratio=0.15):
    """Разделяет train на train и validation с стратификацией."""
    from sklearn.model_selection import train_test_split

    train_split, val_split = train_test_split(
        train_df,
        test_size=val_ratio,
        stratify=train_df['class_index'],
        random_state=42
    )

    # Сохраняем
    train_split_path = os.path.join(PROCESSED_DIR, "train_split.csv")
    val_split_path = os.path.join(PROCESSED_DIR, "val_split.csv")

    train_split.to_csv(train_split_path, index=False)
    val_split.to_csv(val_split_path, index=False)

    print(f"✅ Разделение 70/15/15:")
    print(f"   - Train: {len(train_split):,} записей")
    print(f"   - Val: {len(val_split):,} записей")
    print(f"   - Test: {len(train_df) - len(train_split) - len(val_split):,} записей\n")

    return train_split, val_split


def print_statistics(train_df, test_df):
    """Выводит статистику по датасету."""
    print("📊 Статистика датасета AG News:\n")

    print("Распределение классов в train:")
    class_counts = train_df['class_name'].value_counts().sort_index()
    for class_name, count in class_counts.items():
        pct = count / len(train_df) * 100
        print(f"   {class_name:12}: {count:>6,} ({pct:.1f}%)")

    print(f"\nВсего записей: {len(train_df) + len(test_df):,}")
    print(f"Train: {len(train_df):,}")
    print(f"Test: {len(test_df):,}")

    # Средняя длина текста
    train_df['text_length'] = train_df['title'].str.len()
    print(f"\nСредняя длина текста: {train_df['text_length'].mean():.0f} символов")
    print(f"Медианная длина: {train_df['text_length'].median():.0f} символов")


def main():
    """Главная функция."""
    print("=" * 50)
    print("🚀 Настройка датасета AG News")
    print("=" * 50 + "\n")

    # 1. Установка зависимостей
    install_requirements()

    # 2. Скачивание датасета
    train_df, test_df = download_ag_news()

    # 3. Разделение на train/val
    train_split, val_split = split_train_val(train_df, val_ratio=0.15)

    # 4. Вывод статистики
    print_statistics(train_df, test_df)

    print("\n" + "=" * 50)
    print("✅ Готово! Датасет подготовлен и сохранён в data/")
    print("=" * 50)


if __name__ == "__main__":
    main()
