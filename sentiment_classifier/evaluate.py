from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"text", "label"} - record.keys()
            if missing:
                raise ValueError(f"Line {line_number} is missing: {', '.join(sorted(missing))}")
            if record["label"] not in (0, 1):
                raise ValueError(f"Line {line_number} has a non-binary SST-2 label")
            records.append(record)
    if not records:
        raise ValueError("The test set is empty")
    return records


def _transformer_predictions(model_dir: str, texts: list[str], batch_size: int) -> list[int]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predictions = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**batch).logits
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return predictions


def _lstm_predictions(model_dir: str, texts: list[str], batch_size: int) -> list[int]:
    from torch.nn.utils.rnn import pad_sequence

    from .lstm import SentimentLSTM, tokenize_text

    path = Path(model_dir)
    metadata = json.loads((path / "sentiment_config.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((path / "vocabulary.json").read_text(encoding="utf-8"))
    model = SentimentLSTM(
        metadata["vocabulary_size"], len(metadata["label_names"]),
        metadata["embedding_size"], metadata["hidden_size"],
    )
    model.load_state_dict(torch.load(path / "lstm.pt", map_location="cpu", weights_only=True))
    model.eval()
    max_length = metadata.get("max_length", 256)
    predictions = []
    for start in range(0, len(texts), batch_size):
        sequences = []
        for text in texts[start:start + batch_size]:
            tokens = tokenize_text(text)[:max_length]
            ids = [vocabulary.get(token, 1) for token in tokens] or [1]
            sequences.append(torch.tensor(ids))
        lengths = torch.tensor([len(sequence) for sequence in sequences])
        padded = pad_sequence(sequences, batch_first=True, padding_value=0)
        with torch.no_grad():
            predictions.extend(model(padded, lengths).argmax(dim=-1).tolist())
    return predictions


def evaluate_sst2(model_dir: str, test_file: str, batch_size: int = 32, output: str | None = None):
    path = Path(model_dir)
    metadata_path = path / "sentiment_config.json"
    if not metadata_path.exists():
        raise ValueError(f"Not a trained model directory: {model_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["task_type"] != "single_label" or len(metadata["label_names"]) != 2:
        raise ValueError("The SST-2 challenge evaluator requires a binary single-label model")

    records = read_jsonl(test_file)
    texts = [record["text"] for record in records]
    if metadata["model_type"] == "transformer":
        predictions = _transformer_predictions(model_dir, texts, batch_size)
    else:
        predictions = _lstm_predictions(model_dir, texts, batch_size)

    groups = defaultdict(lambda: [0, 0])
    detailed = []
    for record, prediction in zip(records, predictions):
        correct = int(prediction == record["label"])
        for group in ("overall", f"split:{record.get('split', 'unknown')}",
                      f"category:{record.get('category', 'unknown')}"):
            groups[group][0] += correct
            groups[group][1] += 1
        detailed.append({
            **record,
            "prediction": prediction,
            "predicted_label": metadata["label_names"][prediction],
            "correct": bool(correct),
        })

    summary = {
        group: {"correct": values[0], "total": values[1], "accuracy": values[0] / values[1]}
        for group, values in sorted(groups.items())
    }
    result = {"summary": summary, "examples": detailed}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

