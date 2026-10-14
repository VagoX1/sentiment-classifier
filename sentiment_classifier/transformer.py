from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import multi_label_metrics, single_label_metrics


def train_transformer(
    dataset,
    spec,
    output_dir: str,
    base_model: str = "distilbert/distilbert-base-uncased",
    epochs: float = 3.0,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    seed: int = 42,
):
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    id2label = {index: label for index, label in enumerate(spec.label_names)}
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(spec.label_names),
        id2label=id2label,
        label2id={label: index for index, label in id2label.items()},
        problem_type=(
            "multi_label_classification" if spec.task_type == "multi_label"
            else "single_label_classification"
        ),
    )

    def compute_metrics(prediction):
        logits, labels = prediction
        if spec.task_type == "multi_label":
            return multi_label_metrics(logits, labels)
        return single_label_metrics(logits, labels)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1" if spec.task_type == "multi_label" else "macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    _write_metadata(output_dir, "transformer", spec)
    return metrics


def _write_metadata(output_dir: str, model_type: str, spec, **extra) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_type": model_type,
        "task_type": spec.task_type,
        "label_names": list(spec.label_names),
        **extra,
    }
    (path / "sentiment_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def predict_transformer(model_dir: str, text: str, threshold: float = 0.5):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits[0]

    labels = [model.config.id2label[index] for index in range(model.config.num_labels)]
    if model.config.problem_type == "multi_label_classification":
        probabilities = torch.sigmoid(logits).cpu().tolist()
        selected = [
            {"label": label, "score": float(score)}
            for label, score in zip(labels, probabilities) if score >= threshold
        ]
        return sorted(selected, key=lambda item: item["score"], reverse=True)
    probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
    return sorted(
        ({"label": label, "score": float(score)} for label, score in zip(labels, probabilities)),
        key=lambda item: item["score"], reverse=True,
    )

