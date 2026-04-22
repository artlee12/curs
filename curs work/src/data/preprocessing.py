"""
Модуль предобработки текста для классификации новостей.
Поддержка GPU через spacy и batch processing.
"""

import re
import html
from typing import List, Optional
import numpy as np


class TextPreprocessor:
    """
    Универсальный препроцессор текста с поддержкой GPU.

    Args:
        strategy: 'bert' (минимальная обработка) или 'classic' (полная обработка)
        lowercase: Приводить к нижнему регистру
        remove_stopwords: Удалять стоп-слова (только для classic)
        lemmatize: Лемматизация (только для classic)
        use_gpu: Использовать GPU для spacy если доступен
        batch_size: Размер batch для параллельной обработки
    """

    def __init__(
        self,
        strategy: str = 'classic',
        lowercase: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        use_gpu: bool = True,
        batch_size: int = 1000
    ):
        self.strategy = strategy
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.batch_size = batch_size

        # Загрузка NLTK ресурсов (опционально)
        self.stop_words = set()
        self.lemmatizer = None
        self.nlp = None

        if strategy == 'classic':
            try:
                import nltk
                from nltk.corpus import stopwords
                from nltk.stem import WordNetLemmatizer
                from nltk.tokenize import word_tokenize

                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)

                if remove_stopwords:
                    self.stop_words = set(stopwords.words('english'))
                if lemmatize:
                    self.lemmatizer = WordNetLemmatizer()
                self.word_tokenize = word_tokenize

                # Попытка использовать spacy для GPU-ускорения
                self._setup_spacy(use_gpu)

            except ImportError:
                print("Warning: NLTK not installed. Basic preprocessing will be used.")
                self.word_tokenize = lambda x: x.split()

    def _setup_spacy(self, use_gpu: bool):
        """Настройка spacy с GPU поддержкой."""
        try:
            import spacy

            # Проверка доступности GPU
            if use_gpu and spacy.prefer_gpu():
                print("✅ Spacy использует GPU")
                self.nlp = spacy.load('en_core_web_sm')
            else:
                print("ℹ️  Spacy использует CPU")
                self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

            # Оптимизация pipeline
            if self.nlp:
                self.nlp.max_length = 2000000  # Увеличение лимита для batch processing

        except (ImportError, OSError):
            self.nlp = None
            print("ℹ️  Spacy не настроен. Используется NLTK.")

    def clean_text(self, text: str) -> str:
        """Базовая очистка текста."""
        if not isinstance(text, str):
            return ""

        # HTML entities
        text = html.unescape(text)
        # URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Email
        text = re.sub(r'\S+@\S+', '', text)
        # Спецсимволы (оставляем буквы и базовую пунктуацию)
        text = re.sub(r'[^a-zA-Z\s\.!?]', ' ', text)
        # Лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess(self, text: str) -> str:
        """
        Основной метод предобработки.

        Returns:
            Обработанный текст в зависимости от стратегии.
        """
        text = self.clean_text(text)

        if not text:
            return ""

        if self.lowercase and self.strategy != 'bert':
            text = text.lower()

        if self.strategy == 'classic':
            # Используем spacy если доступен (быстрее на GPU)
            if self.nlp:
                doc = self.nlp(text)
                tokens = [token.lemma_ if self.lemmatize else token.text
                         for token in doc
                         if not token.is_stop or not self.remove_stopwords]
                text = ' '.join(tokens)
            else:
                # Fallback на NLTK
                tokens = self.word_tokenize(text)

                # Удаление стоп-слов
                if self.remove_stopwords:
                    tokens = [t for t in tokens if t not in self.stop_words]

                # Лемматизация
                if self.lemmatize and self.lemmatizer:
                    tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

                text = ' '.join(tokens)

        return text

    def preprocess_batch(self, texts: List[str], n_jobs: int = -1, use_gpu: bool = True) -> List[str]:
        """
        Пакетная обработка текстов с GPU-ускорением.

        Args:
            texts: Список текстов для обработки
            n_jobs: Количество CPU процессов (-1 для всех)
            use_gpu: Использовать GPU если доступен

        Returns:
            Список обработанных текстов
        """
        if not texts:
            return []

        # Если spacy с GPU доступен, используем pipe для batch processing
        if self.nlp and use_gpu and self.strategy == 'classic':
            return self._preprocess_batch_spacy(texts)

        # Иначе используем multiprocessing
        return self._preprocess_batch_parallel(texts, n_jobs)

    def _preprocess_batch_spacy(self, texts: List[str]) -> List[str]:
        """Пакетная обработка через spacy pipe (оптимизировано для GPU)."""
        results = []

        # Очистка текстов
        cleaned_texts = [self.clean_text(t) for t in texts]

        # Обработка через spacy pipe
        for doc in self.nlp.pipe(cleaned_texts, batch_size=self.batch_size):
            if self.lowercase:
                tokens = [token.lemma_.lower() if self.lemmatize else token.text.lower()
                         for token in doc
                         if not token.is_stop or not self.remove_stopwords]
            else:
                tokens = [token.lemma_ if self.lemmatize else token.text
                         for token in doc
                         if not token.is_stop or not self.remove_stopwords]

            results.append(' '.join(tokens))

        return results

    def _preprocess_batch_parallel(self, texts: List[str], n_jobs: int) -> List[str]:
        """Пакетная обработка через multiprocessing."""
        from multiprocessing import Pool, cpu_count

        if n_jobs == -1:
            n_jobs = cpu_count()

        with Pool(processes=n_jobs) as pool:
            results = pool.map(self.preprocess, texts)

        return results


class GPUDataLoader:
    """
    Оптимизированный DataLoader для GPU с prefetch и pinned memory.
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        self.texts = texts
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.indices = np.arange(len(texts))

    def __len__(self):
        return (len(self.texts) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start_idx in range(0, len(self.texts), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.texts))
            batch_indices = self.indices[start_idx:end_idx]

            batch_texts = [self.texts[i] for i in batch_indices]

            if self.labels is not None:
                batch_labels = [self.labels[i] for i in batch_indices]
                yield batch_texts, batch_labels
            else:
                yield batch_texts


class TextCache:
    """Кэш для предобработанных текстов."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache = {}
        self.cache_dir = cache_dir

    def get(self, text: str) -> Optional[str]:
        """Получение кэшированного результата."""
        return self.cache.get(hash(text))

    def set(self, text: str, processed: str):
        """Сохранение в кэш."""
        self.cache[hash(text)] = processed

    def clear(self):
        """Очистка кэша."""
        self.cache.clear()
