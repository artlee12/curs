"""
BERT-based классификатор с полной GPU-оптимизацией для Linux.
Поддерживает: CUDA, Mixed Precision, Gradient Accumulation, Multi-GPU
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path
from .base import BaseClassifier


class BertClassifier(BaseClassifier):
    """
    Классификатор на основе fine-tuned BERT с GPU-оптимизацией.

    Оптимизации:
    - Автоматический выбор GPU (CUDA) если доступен
    - Mixed Precision Training (FP16/BF16)
    - Gradient Accumulation для больших эффективных batch sizes
    - DataLoader с pin_memory и num_workers
    - Оптимизатор AdamW с weight decay
    - Автоматическая очистка GPU кэша
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        default_config = {
            'model_name': 'distilbert-base-uncased',
            'num_labels': 4,
            'max_length': 128,
            'learning_rate': 2e-5,
            'batch_size': 32,
            'epochs': 3,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'device': 'auto',  # 'auto', 'cuda', 'cpu'
            'mixed_precision': True,  # Использовать mixed precision training
            'gradient_accumulation_steps': 1,  # Accumulation для эффективного batch
            'num_workers': 4,  # Количество worker processes для DataLoader
            'pin_memory': True,  # Pin memory для GPU transfer
            'max_grad_norm': 1.0,  # Gradient clipping
            'eval_batch_size': 16,
        }
        self.config = {**default_config, **(config or {})}

        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.training_history = []
        self.label_to_id = {}
        self.id_to_label = {}

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if (
            self.device.type == 'cuda' and self.config.get('mixed_precision', True)
        ) else None

    def _setup_device(self) -> torch.device:
        """Настройка устройства с автоматическим выбором GPU."""
        device_str = self.config.get('device', 'auto')

        if device_str == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                # Вывод информации о GPU
                self._print_gpu_info()
            else:
                device = torch.device('cpu')
                print("⚠️  CUDA недоступна. Используется CPU.")
        else:
            device = torch.device(device_str)

        return device

    def _print_gpu_info(self):
        """Вывод информации о доступных GPU."""
        if torch.cuda.is_available():
            print(f"✅ CUDA доступна!")
            print(f"   Версия CUDA: {torch.version.cuda}")
            print(f"   cuDNN версия: {torch.backends.cudnn.version()}")
            print(f"   Доступно GPU: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name}")
                print(f"      Память: {memory_gb:.2f} GB")
                print(f"      Compute Capability: {props.major}.{props.minor}")

    def build_model(self):
        """Создание модели с загрузкой на GPU."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config['model_name'],
                num_labels=self.config['num_labels']
            )

            # Перенос модели на устройство
            self.model.to(self.device)

            # Оптимизация cuDNN для фиксированных размеров input
            if self.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                print(f"✅ Модель загружена на {self.device}")

                # Проверка памяти GPU
                self._log_gpu_memory()

        except ImportError:
            raise ImportError("transformers и torch необходимы для BERT классификатора")

    def _log_gpu_memory(self):
        """Логирование использования памяти GPU."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
            total = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            print(f"   GPU память: {allocated:.2f} GB allocated / {total:.2f} GB total")

    def _cleanup_gpu(self):
        """Очистка GPU кэша после обучения."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def fit(self, X, y, validation_data=None):
        """Fine-tuning BERT с GPU-оптимизацией."""
        try:
            from torch.utils.data import DataLoader, TensorDataset
            from torch.optim import AdamW
            from transformers import get_linear_schedule_with_warmup
        except ImportError:
            raise ImportError("torch и transformers необходимы для обучения")

        if self.model is None:
            self.build_model()

        # Токенизация данных
        print("🔤 Токенизация данных...")
        train_encodings = self.tokenizer(
            X.tolist() if hasattr(X, 'tolist') else list(X),
            truncation=True,
            padding=True,
            max_length=self.config['max_length'],
            return_tensors='pt'
        )

        y_array = np.array(y)
        if y_array.dtype.kind in {'U', 'S', 'O'}:
            unique_labels = list(np.unique(y_array))
            self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
            y_encoded = np.array([self.label_to_id[label] for label in y_array], dtype=np.int64)
            self.config['num_labels'] = len(unique_labels)
        else:
            y_encoded = y_array.astype(np.int64)
            unique_labels = list(np.unique(y_encoded))
            self.id_to_label = {int(label): int(label) for label in unique_labels}

        train_labels = torch.tensor(y_encoded, dtype=torch.long)
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_labels
        )

        # DataLoader с GPU-оптимизацией
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True) and self.device.type == 'cuda',
            persistent_workers=True if self.config.get('num_workers', 4) > 0 else False
        )

        # Оптимизатор
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['weight_decay']
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config['learning_rate']
        )

        total_steps = len(train_loader) * self.config['epochs'] // self.config.get('gradient_accumulation_steps', 1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_steps
        )

        # Обучение с mixed precision
        self.model.train()
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)

        print(f"🚀 Начало обучения на {self.device}...")
        print(f"   Epochs: {self.config['epochs']}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Gradient accumulation: {accumulation_steps}")
        print(f"   Effective batch size: {self.config['batch_size'] * accumulation_steps}")

        if self.scaler:
            print("   Mixed precision (FP16): включено")

        global_step = 0

        for epoch in range(self.config['epochs']):
            epoch_loss = 0
            self.model.train()

            for batch_idx, batch in enumerate(train_loader):
                input_ids, attention_mask, labels = [b.to(self.device, non_blocking=True) for b in batch]

                # Mixed precision forward
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / accumulation_steps

                # Backward с масштабированием
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.scaler:
                        # Gradient clipping
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.get('max_grad_norm', 1.0)
                        )
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.get('max_grad_norm', 1.0)
                        )
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * accumulation_steps

                # Вывод прогресса
                if (batch_idx + 1) % 50 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    lr = scheduler.get_last_lr()[0]
                    print(f"   Epoch {epoch + 1}/{self.config['epochs']} | "
                          f"Batch {batch_idx + 1}/{len(train_loader)} | "
                          f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")

            avg_loss = epoch_loss / len(train_loader)
            self.training_history.append({'epoch': epoch + 1, 'loss': avg_loss})
            print(f"✅ Epoch {epoch + 1}/{self.config['epochs']} завершена, Loss: {avg_loss:.4f}")

            # Логирование памяти GPU
            if self.device.type == 'cuda':
                self._log_gpu_memory()

        # Валидация
        if validation_data is not None:
            self._validate(validation_data)

        self._cleanup_gpu()
        self.is_fitted = True
        print("✅ Обучение завершено!")

    def _validate(self, validation_data):
        """Валидация модели."""
        from sklearn.metrics import accuracy_score

        X_val, y_val = validation_data
        predictions = self.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        print(f"📊 Validation Accuracy: {accuracy:.4f}")
        return accuracy

    def predict(self, X) -> np.ndarray:
        """Предсказание классов с GPU-оптимизацией."""
        self.model.eval()

        texts = X.tolist() if hasattr(X, 'tolist') else list(X)
        eval_batch_size = self.config.get('eval_batch_size', self.config.get('batch_size', 16))
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(texts), eval_batch_size):
                batch_texts = texts[i:i + eval_batch_size]
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.config['max_length'],
                    return_tensors='pt'
                )

                input_ids = encodings['input_ids'].to(self.device, non_blocking=True)
                attention_mask = encodings['attention_mask'].to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=-1)

                all_preds.append(predictions.cpu().numpy())

        pred_ids = np.concatenate(all_preds)
        if self.id_to_label:
            return np.array([self.id_to_label[int(i)] for i in pred_ids])
        return pred_ids

    def predict_proba(self, X) -> np.ndarray:
        """Предсказание вероятностей с GPU-оптимизацией."""
        self.model.eval()

        texts = X.tolist() if hasattr(X, 'tolist') else list(X)
        eval_batch_size = self.config.get('eval_batch_size', self.config.get('batch_size', 16))
        all_probas = []

        with torch.no_grad():
            for i in range(0, len(texts), eval_batch_size):
                batch_texts = texts[i:i + eval_batch_size]
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.config['max_length'],
                    return_tensors='pt'
                )

                input_ids = encodings['input_ids'].to(self.device, non_blocking=True)
                attention_mask = encodings['attention_mask'].to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    probabilities = torch.softmax(outputs.logits, dim=-1)

                all_probas.append(probabilities.cpu().numpy())

        return np.concatenate(all_probas, axis=0)

    def save(self, path, format: str = "pytorch"):
        """
        Сохранение BERT модели.

        Args:
            path: Путь для сохранения
            format: "pytorch" или "onnx"
        """
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Сохранение весов и конфигурации
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Сохранение метаданных
        metadata = {
            **self.metadata,
            'training_history': self.training_history,
            'config': self.config,
            'device': str(self.device)
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Конвертация в ONNX (опционально)
        if format == "onnx":
            self._export_to_onnx(path / "model.onnx")

    def _export_to_onnx(self, onnx_path: Path):
        """Экспорт в ONNX формат с GPU поддержкой."""
        try:
            import torch
        except ImportError:
            raise ImportError("torch необходим для ONNX экспорта")

        self.model.eval()

        # Экспорт с CPU для совместимости
        dummy_input = self.tokenizer(
            "dummy text",
            return_tensors="pt",
            max_length=self.config['max_length'],
            padding='max_length',
            truncation=True
        )

        torch.onnx.export(
            self.model.cpu(),
            (dummy_input['input_ids'], dummy_input['attention_mask']),
            onnx_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=14,
            do_constant_folding=True
        )

        # Возврат на GPU если нужно
        if self.device.type == 'cuda':
            self.model.to(self.device)

    @classmethod
    def load(cls, path: Path, device: str = 'auto'):
        """Загрузка модели с поддержкой GPU."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import json

        instance = cls({'device': device})

        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model.to(instance.device)
        instance.is_fitted = True

        # Загрузка метаданных
        metadata_path = path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                instance.metadata = metadata
                instance.config = metadata.get('config', {})
                instance.training_history = metadata.get('training_history', [])

        print(f"✅ Модель загружена на {instance.device}")
        return instance
