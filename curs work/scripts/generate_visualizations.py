#!/usr/bin/env python3
"""Generate comparison visualizations for coursework report."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

from src.models.bert_classifier import BertClassifier
from src.utils.config import load_config


def setup_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.facecolor"] = "#f7f7f5"
    plt.rcParams["axes.facecolor"] = "#fcfcfb"
    plt.rcParams["axes.edgecolor"] = "#d9d9d6"
    plt.rcParams["savefig.facecolor"] = "#f7f7f5"


def ensure_output_dir() -> Path:
    out = Path("artifacts/visualizations_20260429")
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_split():
    config = load_config("configs/bert_config.yaml")
    df = pd.read_csv(Path(config.data.raw_path) / config.data.train_file)
    if "title" in df.columns and "description" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    elif "title" in df.columns and "text" not in df.columns:
        df["text"] = df["title"]

    if "class_name" in df.columns and "class" not in df.columns:
        df["class"] = df["class_name"]
    elif "class_index" in df.columns and "class" not in df.columns:
        df["class"] = df["class_index"]

    X = df["text"]
    y = df["class"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def render_confusion(pred_path: Path, model_name: str, out_dir: Path):
    df = pd.read_csv(pred_path)
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="YlGnBu", ax=ax, colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix: {model_name}", fontweight="bold", pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_dir / f"cm_{model_name.lower().replace(' ', '_')}.png", dpi=180)
    plt.close(fig)


def render_tsne(bert: BertClassifier, X_test: pd.Series, y_test: pd.Series, out_dir: Path):
    sample_n = min(1500, len(X_test))
    sample_idx = np.random.RandomState(42).choice(len(X_test), sample_n, replace=False)
    texts = X_test.iloc[sample_idx].tolist()
    labels = y_test.iloc[sample_idx].values

    tokenizer = bert.tokenizer
    model = bert.model
    device = bert.device

    embs = []
    batch_size = 32
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            cls = outputs.last_hidden_state[:, 0, :]
            embs.append(cls.cpu().numpy())

    emb_mat = np.vstack(embs)
    tsne = TSNE(n_components=2, random_state=42, perplexity=35, init="pca")
    proj = tsne.fit_transform(emb_mat)

    plot_df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1], "label": labels})
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="label",
        palette="Set2",
        alpha=0.85,
        s=36,
        linewidth=0,
        ax=ax,
    )
    ax.set_title("t-SNE of BERT [CLS] Embeddings", fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    ax.legend(title="Class", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "tsne_bert_embeddings.png", dpi=180)
    plt.close(fig)


def render_attention_example(bert: BertClassifier, out_dir: Path):
    text = "Apple unveils a new AI chip for smartphones as competitors race in mobile machine learning."
    tokenizer = bert.tokenizer
    model = bert.model
    device = bert.device
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    if not outputs.attentions:
        raise RuntimeError("Attention tensors are unavailable for current model configuration")
    attn = outputs.attentions[-1][0]  # last layer, batch 0: [heads, seq, seq]
    attn_mean = attn.mean(dim=0).cpu().numpy()

    keep = min(12, len(tokens))
    tokens = tokens[:keep]
    attn_mean = attn_mean[:keep, :keep]

    # Make token labels cleaner for plotting
    clean_tokens = [t.replace("##", "") for t in tokens]

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        attn_mean,
        xticklabels=clean_tokens,
        yticklabels=clean_tokens,
        cmap="mako",
        cbar=True,
        square=True,
        ax=ax,
    )
    ax.set_title("BERT Attention Heatmap (Last Layer Mean Heads)", fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelrotation=90, labelsize=10)
    ax.tick_params(axis="y", labelrotation=0, labelsize=10)
    fig.subplots_adjust(left=0.22, bottom=0.28, right=0.96, top=0.92)
    fig.savefig(out_dir / "attention_heatmap_example.png", dpi=180)
    plt.close(fig)


def main():
    setup_style()
    out_dir = ensure_output_dir()

    render_confusion(
        Path("artifacts/20260424_100921/predictions/bert_predictions.csv"),
        "BERT",
        out_dir,
    )
    render_confusion(
        Path("artifacts/20260424_112123/predictions/cnn_predictions.csv"),
        "CNN",
        out_dir,
    )
    render_confusion(
        Path("artifacts/stacking_pretrained_20260429/predictions/stacking_pretrained_predictions.csv"),
        "Stacking",
        out_dir,
    )

    X_test, y_test = load_split()
    bert = BertClassifier.load(Path("models/serialized/bert_20260424_100921"), device="auto")
    render_tsne(bert, X_test, y_test, out_dir)
    render_attention_example(bert, out_dir)

    print(f"Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
