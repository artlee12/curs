#!/usr/bin/env python3
"""
check_gpu.py - Проверка GPU и рекомендации по настройке.
Запуск перед обучением для оптимальных параметров.
"""

import sys
import torch
import subprocess


def check_gpu():
    """Проверка доступности GPU."""
    print("="*60)
    print("🔍 ПРОВЕРКА GPU ДЛЯ ОБУЧЕНИЯ МОДЕЛЕЙ")
    print("="*60)

    # PyTorch информация
    print(f"\n📦 PyTorch:")
    print(f"   Версия: {torch.__version__}")
    print(f"   CUDA доступна: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   Версия CUDA: {torch.version.cuda}")
        print(f"   Версия cuDNN: {torch.backends.cudnn.version()}")
        print(f"   Доступно GPU: {torch.cuda.device_count()}")

        # Детали GPU
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)

            print(f"\n🖥️  GPU {i}: {props.name}")
            print(f"   Архитектура: {props.major}.{props.minor}")
            print(f"   Память: {memory_gb:.2f} GB")
            print(f"   Мультипроцессоры: {props.multi_processor_count}")

            # Рекомендации по параметрам
            print(f"\n💡 РЕКОМЕНДАЦИИ ДЛЯ BERT:")

            if memory_gb >= 24:
                print("   Большая память GPU - можно использовать:")
                print("   --batch-size 128")
                print("   --epochs 5")
                print("   Модели: bert-base, bert-large, roberta-large")
            elif memory_gb >= 16:
                print("   Хорошая память GPU - рекомендуется:")
                print("   --batch-size 64")
                print("   --epochs 3-5")
                print("   Модели: bert-base, distilbert")
            elif memory_gpu >= 8:
                print("   Средняя память GPU - рекомендуется:")
                print("   --batch-size 32")
                print("   --epochs 3")
                print("   Модели: distilbert-base")
            else:
                print("   Ограниченная память GPU:")
                print("   --batch-size 16")
                print("   --epochs 3")
                print("   Используйте gradient_accumulation_steps=4")

            # Проверка выделенной памяти
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"\n📊 Текущее использование памяти GPU {i}:")
            print(f"   Allocated: {allocated:.2f} GB")
            print(f"   Reserved:  {reserved:.2f} GB")

        # Проверка cuML
        print(f"\n📦 RAPIDS AI (cuML для ускорения ML):")
        try:
            import cuml
            print("   ✅ cuML установлен - Logistic Regression будет на GPU")
        except ImportError:
            print("   ⚠️  cuML не установлен")
            print("   Установка: pip install cuml-cu11")

        # Проверка spacy GPU
        print(f"\n📦 Spacy:")
        try:
            import spacy
            spacy.prefer_gpu()
            print("   ✅ Spacy может использовать GPU")
        except:
            print("   ⚠️  Spacy использует CPU")

        print("\n" + "="*60)
        print("✅ GPU ГОТОВ К ОБУЧЕНИЮ")
        print("="*60)
        print("\nПример запуска обучения:")
        print("  python scripts/train.py --model bert --epochs 3")
        print("\n")
        return True

    else:
        print("\n" + "="*60)
        print("⚠️  GPU НЕДОСТУПНА")
        print("="*60)
        print("\nОбучение будет на CPU (медленнее)")
        print("\nДля GPU установите PyTorch с CUDA:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("\nУбедитесь что установлены:")
        print("  - NVIDIA драйверы: https://www.nvidia.com/drivers")
        print("  - CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("\n")
        return False


def test_gpu_training():
    """Тестовое обучение на GPU."""
    if not torch.cuda.is_available():
        return

    print("🧪 Тестовое вычисление на GPU...")

    # Создаем тензоры на GPU
    device = torch.device('cuda')

    # Тестовая матричная операция
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)

    import time
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"   Матрица 1000x1000 умножение: {elapsed*1000:.2f} мс")
    print(f"   ✅ GPU работает корректно\n")


def print_commands():
    """Вывод полезных команд."""
    print("📋 ПОЛЕЗНЫЕ КОМАНДЫ:")
    print("-" * 60)
    print("Проверка GPU в системе:")
    print("  nvidia-smi")
    print("\nОбучение моделей:")
    print("  python scripts/train.py --model tfidf_logreg")
    print("  python scripts/train.py --model bert --epochs 3")
    print("  python scripts/train.py --model bert --no-gpu  # CPU")
    print("\nМониторинг в реальном времени:")
    print("  watch -n 1 nvidia-smi")
    print("-" * 60)


if __name__ == "__main__":
    gpu_available = check_gpu()

    if gpu_available:
        test_gpu_training()

    print_commands()

    # Статус выхода
    sys.exit(0 if gpu_available else 1)
