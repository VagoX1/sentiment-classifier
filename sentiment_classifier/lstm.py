from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from .metrics import multi_label_metrics, single_label_metrics
from .transformer import _write_metadata


TOKEN_PATTERN = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(texts, min_frequency: int = 2, max_size: int = 50_000):
    counts = Counter(token for text in texts for token in tokenize_text(text))
    tokens = [token for token, count in counts.most_common() if count >= min_frequency]
    tokens = tokens[: max(0, max_size - 2)]
    return {"<pad>": 0, "<unk>": 1, **{token: i + 2 for i, token in enumerate(tokens)}}


class TextDataset(Dataset):
    def __init__(self, split, vocabulary, multi_label: bool, max_length: int = 256):
        self.split = split
        self.vocabulary = vocabulary
        self.multi_label = multi_label
        self.max_length = max_length

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        item = self.split[index]
        tokens = tokenize_text(item["text"])[: self.max_length]
        ids = [self.vocabulary.get(token, 1) for token in tokens] or [1]
        label_dtype = torch.float32 if self.multi_label else torch.long
        return torch.tensor(ids), torch.tensor(item["labels"], dtype=label_dtype)


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, lengths, torch.stack(labels)


class SentimentLSTM(nn.Module):
    def __init__(self, vocabulary_size: int, num_labels: int, embedding_size=128, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, token_ids, lengths):
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(representation))


def train_lstm(
    dataset,
    spec,
    output_dir: str,
    epochs: float = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_length: int = 256,
    seed: int = 42,
):
    torch.manual_seed(seed)
    vocabulary = build_vocabulary(dataset["train"]["text"])
    is_multi = spec.task_type == "multi_label"
    train_data = TextDataset(dataset["train"], vocabulary, is_multi, max_length)
    validation_data = TextDataset(dataset["validation"], vocabulary, is_multi, max_length)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentLSTM(len(vocabulary), len(spec.label_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss() if is_multi else nn.CrossEntropyLoss()

    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        for token_ids, lengths, labels in train_loader:
            optimizer.zero_grad()
            logits = model(token_ids.to(device), lengths)
            loss = criterion(logits, labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        metrics = evaluate_lstm(model, validation_loader, device, is_multi)
        print(f"epoch={epoch + 1} loss={total_loss / max(1, len(train_loader)):.4f} metrics={metrics}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "lstm.pt")
    (output / "vocabulary.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    _write_metadata(
        output_dir, "lstm", spec, vocabulary_size=len(vocabulary),
        embedding_size=128, hidden_size=128, max_length=max_length,
    )
    return evaluate_lstm(model, validation_loader, device, is_multi)


def evaluate_lstm(model, loader, device, multi_label: bool):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for token_ids, lengths, labels in loader:
            all_logits.append(model(token_ids.to(device), lengths).cpu())
            all_labels.append(labels)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return multi_label_metrics(logits, labels) if multi_label else single_label_metrics(logits, labels)


def predict_lstm(model_dir: str, text: str, threshold: float = 0.5):
    path = Path(model_dir)
    metadata = json.loads((path / "sentiment_config.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((path / "vocabulary.json").read_text(encoding="utf-8"))
    model = SentimentLSTM(
        metadata["vocabulary_size"], len(metadata["label_names"]),
        metadata["embedding_size"], metadata["hidden_size"],
    )
    model.load_state_dict(torch.load(path / "lstm.pt", map_location="cpu", weights_only=True))
    model.eval()
    tokens = tokenize_text(text)[: metadata.get("max_length", 256)]
    ids = [vocabulary.get(token, 1) for token in tokens] or [1]
    token_ids = torch.tensor([ids])
    with torch.no_grad():
        logits = model(token_ids, torch.tensor([len(ids)]))[0]

    if metadata["task_type"] == "multi_label":
        scores = torch.sigmoid(logits).tolist()
        results = [
            {"label": label, "score": float(score)}
            for label, score in zip(metadata["label_names"], scores) if score >= threshold
        ]
    else:
        scores = torch.softmax(logits, dim=-1).tolist()
        results = [
            {"label": label, "score": float(score)}
            for label, score in zip(metadata["label_names"], scores)
        ]
    return sorted(results, key=lambda item: item["score"], reverse=True)
