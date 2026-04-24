#!/bin/bash
# setup_and_download.sh - Полная настройка окружения и датасета для Linux с GPU

set -e  # Остановка при ошибке

echo "=========================================="
echo "🚀 Настройка окружения для GPU ML"
echo "=========================================="

# Переходим в директорию проекта
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "📁 Директория проекта: $PROJECT_DIR"

# Проверка Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  Этот скрипт оптимизирован для Linux. Текущая ОС: $OSTYPE"
    read -p "Продолжить? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Проверка наличия python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python 3.10+"
    echo "   sudo apt-get install python3 python3-pip python3-venv"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "🐍 Python версия: $PYTHON_VERSION"

# Проверка наличия CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "✅ CUDA найдена! Версия: $CUDA_VERSION"
else
    echo "⚠️  CUDA не найдена. Для GPU ускорения установите CUDA Toolkit:"
    echo "   https://developer.nvidia.com/cuda-downloads"
fi

# Проверка NVIDIA драйверов
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA драйверы установлены:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | while read line; do
        echo "   GPU: $line"
    done
else
    echo "⚠️  nvidia-smi не найден. Установите NVIDIA драйверы для GPU."
fi

# 1. Создаём виртуальное окружение
echo ""
echo "📦 Шаг 1: Создание виртуального окружения..."
if [ ! -d "$PROJECT_DIR/venv" ]; then
    python3 -m venv venv
    echo "✅ venv создан"
else
    echo "✅ venv уже существует"
fi

# 2. Активируем окружение
echo ""
echo "📦 Шаг 2: Активация окружения..."
source "$PROJECT_DIR/venv/bin/activate"
echo "✅ Окружение активировано"

# 3. Устанавливаем зависимости
echo ""
echo "📦 Шаг 3: Установка зависимостей..."
pip install --upgrade pip -q

# PyTorch с CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "📦 Установка PyTorch с CUDA поддержкой..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
else
    echo "📦 Установка PyTorch (CPU версия)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
fi

# Установка RAPIDS/cuML для Linux с GPU
if [[ "$OSTYPE" == "linux-gnu"* ]] && command -v nvidia-smi &> /dev/null; then
    echo "📦 Установка RAPIDS AI (cuML, cuDF) для GPU ускорения..."
    pip install cuml-cu11 cudf-cu11 cupy-cuda11x -q || echo "⚠️  Не удалось установить RAPIDS (опционально)"
fi

# Остальные зависимости
echo "📦 Установка остальных зависимостей..."
pip install -r requirements.txt -q || pip install -r requirements.txt --no-deps -q

# Установка spacy моделей
echo "📦 Загрузка моделей Spacy..."
python3 -m spacy download en_core_web_sm -q || echo "⚠️  Не удалось загрузить spacy модель"

echo "✅ Все зависимости установлены"

# 4. Проверка GPU
echo ""
echo "🔍 Шаг 4: Проверка GPU..."
python3 << 'EOF'
import torch
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA версия: {torch.version.cuda}")
    print(f"Количество GPU: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"   Память: {props.total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  GPU не доступна. Обучение будет на CPU.")
EOF

# 5. Запускаем скрипт загрузки датасета
echo ""
echo "📥 Шаг 5: Загрузка датасета AG News..."
python3 scripts/setup_dataset.py

echo ""
echo "=========================================="
echo "✅ Всё готово!"
echo "=========================================="
echo ""
echo "Для активации окружения в будущем:"
echo "  source venv/bin/activate"
echo ""
echo "Для обучения с GPU:"
echo "  python scripts/train.py --model bert"
echo ""
echo "Данные находятся в:"
echo "  - data/raw/train.csv"
echo "  - data/raw/test.csv"
echo ""

# Проверка установки
echo "🔍 Итоговая проверка..."
python3 << 'EOF'
print("\n" + "="*50)
print("ПРОВЕРКА УСТАНОВКИ")
print("="*50)

modules = [
    ("torch", "PyTorch"),
    ("transformers", "Transformers"),
    ("sklearn", "Scikit-learn"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("spacy", "Spacy"),
]

all_ok = True
for module, name in modules:
    try:
        __import__(module)
        print(f"✅ {name}: OK")
    except ImportError:
        print(f"❌ {name}: НЕ УСТАНОВЛЕН")
        all_ok = False

# Проверка cuML
try:
    import cuml
    print(f"✅ cuML (GPU): OK")
except ImportError:
    print(f"⚠️  cuML (GPU): Не установлен (опционально)")

print("="*50)
if all_ok:
    print("✅ Все основные модули установлены!")
else:
    print("❌ Некоторые модули не установлены")
print("="*50 + "\n")
EOF
