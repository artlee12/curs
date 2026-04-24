# Dockerfile для курсовой работы с GPU ускорением
# Базовый образ с CUDA и cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Отключение интерактивного режима apt
ENV DEBIAN_FRONTEND=noninteractive

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование файлов проекта
COPY requirements.txt .
COPY configs/ ./configs/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Установка Python зависимостей
RUN pip3 install --upgrade pip setuptools wheel

# PyTorch с CUDA 11.8
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# RAPIDS AI для GPU ML
RUN pip3 install cuml-cu11 cudf-cu11 cupy-cuda11x || echo "RAPIDS installation optional"

# Остальные зависимости
RUN pip3 install -r requirements.txt || pip3 install transformers scikit-learn pandas numpy spacy

# Загрузка модели Spacy
RUN python3 -m spacy download en_core_web_sm

# Установка переменных окружения для GPU
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TOKENIZERS_PARALLELISM=false

# Проверка установки
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Точка входа
ENTRYPOINT ["python3", "scripts/train.py"]

# Команда по умолчанию
CMD ["--model", "bert", "--epochs", "3"]
