"""
CNN классификатор текстов с Word2Vec-style эмбеддингами.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseClassifier


class _TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        num_filters: int,
        kernel_sizes: List[int],
        dropout: float,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)

        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            p = torch.max(c, dim=2).values
            pooled.append(p)

        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        return self.fc(x)


class Word2VecCNNClassifier(BaseClassifier):
    """TextCNN с trainable Word2Vec-style эмбеддингами."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, use_gpu: bool = True):
        super().__init__(config)

        default_config = {
            "embedding_dim": 200,
            "max_length": 80,
            "min_freq": 2,
            "batch_size": 256 if use_gpu and torch.cuda.is_available() else 64,
            "epochs": 5,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "dropout": 0.3,
            "num_filters": 128,
            "kernel_sizes": [2, 3, 4],
            "num_workers": 2,
            "pin_memory": use_gpu,
            "eval_batch_size": 512,
        }

        self.config = {**default_config, **(config or {})}
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        self.model: Optional[_TextCNN] = None
        self.training_history: List[Dict[str, float]] = []
        self.word_to_idx: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self.idx_to_word: Dict[int, str] = {0: "<pad>", 1: "<unk>"}
        self.label_to_id: Dict[Any, int] = {}
        self.id_to_label: Dict[int, Any] = {}

    def _tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        return text.split()

    def _build_vocab(self, texts: List[str]):
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        for token, freq in counter.items():
            if freq >= self.config["min_freq"] and token not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[token] = idx
                self.idx_to_word[idx] = token

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        max_len = self.config["max_length"]
        unk = self.word_to_idx["<unk>"]
        pad = self.word_to_idx["<pad>"]

        encoded = np.full((len(texts), max_len), pad, dtype=np.int64)
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)[:max_len]
            ids = [self.word_to_idx.get(tok, unk) for tok in tokens]
            if ids:
                encoded[i, : len(ids)] = ids
        return encoded

    def _encode_labels(self, y) -> np.ndarray:
        y_array = np.array(y)
        if not self.label_to_id:
            unique_labels = sorted(np.unique(y_array).tolist())
            self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
            self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        return np.array([self.label_to_id[label] for label in y_array], dtype=np.int64)

    def fit(self, X, y, validation_data=None):
        X_list = X.tolist() if hasattr(X, "tolist") else list(X)
        y_encoded = self._encode_labels(y)

        print("🧠 Подготовка Word2Vec-style словаря...")
        self._build_vocab(X_list)

        print("🔢 Кодирование текстов...")
        X_encoded = self._encode_texts(X_list)

        self.model = _TextCNN(
            vocab_size=len(self.word_to_idx),
            embedding_dim=self.config["embedding_dim"],
            num_classes=len(self.label_to_id),
            num_filters=self.config["num_filters"],
            kernel_sizes=self.config["kernel_sizes"],
            dropout=self.config["dropout"],
        ).to(self.device)

        dataset = TensorDataset(
            torch.from_numpy(X_encoded),
            torch.from_numpy(y_encoded),
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"] and self.device.type == "cuda",
            persistent_workers=self.config["num_workers"] > 0,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        print(f"🚀 Обучение CNN на {self.device}...")
        for epoch in range(self.config["epochs"]):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_seen = 0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == batch_y).sum().item()
                total_seen += batch_x.size(0)

            epoch_loss = total_loss / max(total_seen, 1)
            epoch_acc = total_correct / max(total_seen, 1)
            self.training_history.append(
                {"epoch": epoch + 1, "loss": float(epoch_loss), "accuracy": float(epoch_acc)}
            )
            print(
                f"   Epoch {epoch + 1}/{self.config['epochs']} | "
                f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
            )

        if validation_data is not None:
            from sklearn.metrics import accuracy_score

            X_val, y_val = validation_data
            val_pred = self.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"📊 Validation Accuracy: {val_acc:.4f}")

        self.metadata.update(
            {
                "vocab_size": len(self.word_to_idx),
                "num_classes": len(self.label_to_id),
                "device": str(self.device),
                "training_history": self.training_history,
                "config": self.config,
            }
        )
        self.is_fitted = True

    def _predict_logits(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не обучена")

        self.model.eval()
        X_list = X.tolist() if hasattr(X, "tolist") else list(X)
        X_encoded = self._encode_texts(X_list)
        x_tensor = torch.from_numpy(X_encoded)

        loader = DataLoader(
            x_tensor,
            batch_size=self.config["eval_batch_size"],
            shuffle=False,
            num_workers=0,
        )

        logits_out = []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                logits = self.model(batch_x)
                logits_out.append(logits.cpu().numpy())

        return np.concatenate(logits_out, axis=0)

    def predict(self, X) -> np.ndarray:
        logits = self._predict_logits(X)
        pred_ids = np.argmax(logits, axis=1)
        return np.array([self.id_to_label[int(i)] for i in pred_ids])

    def predict_proba(self, X) -> np.ndarray:
        logits = self._predict_logits(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def save(self, path, format: str = "pytorch"):
        if self.model is None:
            raise RuntimeError("Модель не обучена")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path / "model.pt")
        payload = {
            "config": self.config,
            "word_to_idx": self.word_to_idx,
            "label_to_id": self.label_to_id,
            "metadata": self.metadata,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(payload, f, indent=2)
