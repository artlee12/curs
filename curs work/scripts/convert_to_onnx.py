#!/usr/bin/env python3
"""
Конвертация моделей в ONNX формат для оптимизированного инференса.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bert_classifier import BertClassifier


def convert_bert_to_onnx(model_path: str, output_path: str):
    """Конвертация BERT модели в ONNX."""
    print(f"Loading model from {model_path}...")
    model = BertClassifier.load(model_path)

    print(f"Exporting to ONNX: {output_path}...")
    model.save(output_path, format="onnx")
    print("Conversion completed!")


def main():
    parser = argparse.ArgumentParser(description="Convert models to ONNX")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True,
                       choices=["bert"])
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    if args.model_type == "bert":
        convert_bert_to_onnx(args.model_path, args.output)
    else:
        print(f"ONNX conversion for {args.model_type} not implemented")


if __name__ == "__main__":
    main()
