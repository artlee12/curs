"""
Менеджер артефактов для сохранения метрик, предсказаний и визуализаций.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import matplotlib
matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class ArtifactManager:
    """
    Централизованное управление артефактами эксперимента.
    """

    def __init__(self, experiment_id: str, base_path: str = "./artifacts"):
        self.experiment_id = experiment_id
        self.base_path = Path(base_path).resolve()  # Абсолютный путь
        self.experiment_path = self.base_path / experiment_id
        self._create_directories()

        self.artifacts_log = []

    def _create_directories(self):
        """Создание структуры директорий."""
        for subdir in ['metrics', 'predictions', 'confusion_matrices', 'visualizations', 'models']:
            (self.experiment_path / subdir).mkdir(parents=True, exist_ok=True)

    def save_metrics(self, metrics: Dict[str, float], model_name: str):
        """
        Сохранение метрик в JSON.

        Args:
            metrics: Словарь метрик
            model_name: Название модели
        """
        metrics_data = {
            "experiment_id": self.experiment_id,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }

        filepath = self.experiment_path / "metrics" / f"{model_name}_metrics.json"
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        self.artifacts_log.append({
            "type": "metrics",
            "model": model_name,
            "path": str(filepath)
        })

    def save_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        model_name: str,
        text_samples: Optional[List[str]] = None
    ):
        """
        Сохранение предсказаний в CSV.

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            y_proba: Вероятности классов (опционально)
            model_name: Название модели
            text_samples: Исходные тексты (опционально)
        """
        df_data = {
            'y_true': y_true,
            'y_pred': y_pred,
            'correct': y_true == y_pred
        }

        if y_proba is not None:
            for i in range(y_proba.shape[1]):
                df_data[f'proba_class_{i}'] = y_proba[:, i]

        if text_samples:
            df_data['text'] = text_samples

        df = pd.DataFrame(df_data)

        filepath = self.experiment_path / "predictions" / f"{model_name}_predictions.csv"
        df.to_csv(filepath, index=False)

        self.artifacts_log.append({
            "type": "predictions",
            "model": model_name,
            "path": str(filepath),
            "num_samples": len(y_true),
            "accuracy": (y_true == y_pred).mean()
        })

    def save_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        model_name: str
    ):
        """
        Сохранение confusion matrix как изображения.

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            class_names: Названия классов
            model_name: Название модели
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        filepath = self.experiment_path / "confusion_matrices" / f"{model_name}_cm.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        # Также сохраняем как CSV
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(self.experiment_path / "confusion_matrices" / f"{model_name}_cm.csv")

        self.artifacts_log.append({
            "type": "confusion_matrix",
            "model": model_name,
            "path": str(filepath)
        })

    def save_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        model_name: str
    ):
        """Сохранение classification report."""
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True
        )

        filepath = self.experiment_path / "metrics" / f"{model_name}_report.json"
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

    def save_training_curves(
        self,
        history: List[Dict[str, Any]],
        model_name: str
    ):
        """Сохранение графиков обучения."""
        epochs = [h['epoch'] for h in history]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curve
        if 'loss' in history[0]:
            losses = [h['loss'] for h in history]
            axes[0].plot(epochs, losses, 'b-', label='Train Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()

        # Accuracy curve
        if 'accuracy' in history[0]:
            accuracies = [h['accuracy'] for h in history]
            axes[1].plot(epochs, accuracies, 'g-', label='Train Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training Accuracy')
            axes[1].legend()

        plt.tight_layout()

        filepath = self.experiment_path / "visualizations" / f"{model_name}_training_curves.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        self.artifacts_log.append({
            "type": "training_curves",
            "model": model_name,
            "path": str(filepath)
        })

    def save_model_info(self, model_path: str, model_name: str, metadata: Dict):
        """Сохранение информации о модели."""
        info = {
            "experiment_id": self.experiment_id,
            "model_name": model_name,
            "model_path": model_path,
            "metadata": metadata,
            "saved_at": datetime.now().isoformat()
        }

        filepath = self.experiment_path / "models" / f"{model_name}_info.json"
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Получение сводки по всем артефактам."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_path": str(self.experiment_path),
            "total_artifacts": len(self.artifacts_log),
            "artifacts": self.artifacts_log
        }

    def save_summary(self):
        """Сохранение сводки эксперимента."""
        summary = self.get_summary()
        filepath = self.experiment_path / "experiment_summary.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
