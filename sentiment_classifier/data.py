"""Dataset definitions, label normalization, and custom training-data mixing."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Literal


TaskType = Literal["single_label", "multi_label"]


@dataclass(frozen=True)
class DatasetSpec:
    """Describe source columns and the ordered label IDs used by both backends."""

    repo: str
    config: str | None
    text_column: str
    label_column: str
    label_names: tuple[str, ...]
    task_type: TaskType = "single_label"
    description: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "sst2": DatasetSpec(
        "stanfordnlp/sst2", None, "sentence", "label",
        ("negative", "positive"), description="Short English movie-review phrases",
    ),
    "imdb": DatasetSpec(
        "stanfordnlp/imdb", None, "text", "label",
        ("negative", "positive"), description="Long English movie reviews",
    ),
    "tweet_eval": DatasetSpec(
        "cardiffnlp/tweet_eval", "sentiment", "text", "label",
        ("negative", "neutral", "positive"), description="English social-media sentiment",
    ),
    "amazon_polarity": DatasetSpec(
        "mteb/amazon_polarity", None, "text", "label",
        ("negative", "positive"), description="English product reviews",
    ),
    "yelp": DatasetSpec(
        "Yelp/yelp_review_full", None, "text", "label",
        ("1_star", "2_stars", "3_stars", "4_stars", "5_stars"),
        description="English business reviews scored from one to five stars",
    ),
    "financial_phrasebank": DatasetSpec(
        "atrost/financial_phrasebank", None, "sentence", "label",
        ("negative", "neutral", "positive"), description="English financial-news phrases",
    ),
    "go_emotions": DatasetSpec(
        "google-research-datasets/go_emotions", "simplified", "text", "labels",
        (
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral",
        ),
        task_type="multi_label",
        description="Fine-grained, potentially multi-label emotions in Reddit comments",
    ),
}


def list_datasets() -> str:
    """Format the dataset registry for the CLI without downloading any data."""
    rows = []
    for name, spec in DATASETS.items():
        rows.append(
            f"{name:22} {len(spec.label_names):2} labels  "
            f"{spec.task_type:12}  {spec.description}"
        )
    return "\n".join(rows)


def load_and_prepare(name: str, max_train_samples: int | None = None, seed: int = 42):
    """Download a Hub dataset and normalize it to text + labels columns."""
    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from: {', '.join(DATASETS)}")
    spec = DATASETS[name]
    args = [spec.repo] + ([spec.config] if spec.config else [])
    raw = load_dataset(*args)

    if "train" not in raw:
        raise ValueError(f"{spec.repo} does not expose a train split")
    # Derive validation from training only; preserve any existing test split.
    if "validation" not in raw:
        split = raw["train"].train_test_split(test_size=0.1, seed=seed)
        raw = DatasetDict({**raw, "train": split["train"], "validation": split["test"]})

    def normalize(example):
        result = {"text": example[spec.text_column]}
        source = example[spec.label_column]
        if spec.task_type == "multi_label":
            # Multi-hot targets allow several emotions to be active at once.
            vector = [0.0] * len(spec.label_names)
            for label_id in source:
                vector[int(label_id)] = 1.0
            result["labels"] = vector
        else:
            result["labels"] = int(source)
        return result

    normalized = DatasetDict()
    for split_name, split in raw.items():
        # SST-2's public test labels are hidden (-1), so it remains available for
        # prediction but validation is the evaluation split.
        prepared = split.map(normalize, remove_columns=split.column_names)
        if split_name == "train" and max_train_samples is not None:
            count = min(max_train_samples, len(prepared))
            prepared = prepared.shuffle(seed=seed).select(range(count))
        normalized[split_name] = prepared
    return normalized, spec


def parse_external_label(value, label_names: tuple[str, ...]) -> int:
    """Accept integer IDs, numeric strings, or label names from custom files."""
    if isinstance(value, bool):
        raise ValueError("Boolean labels are not supported")
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        label_id = int(value)
    else:
        normalized = str(value).strip().lower()
        names = [name.lower() for name in label_names]
        if normalized not in names:
            raise ValueError(f"Unknown label {value!r}; expected an ID or one of {label_names}")
        label_id = names.index(normalized)
    if not 0 <= label_id < len(label_names):
        raise ValueError(f"Label ID {label_id} is outside 0..{len(label_names) - 1}")
    return label_id


def _read_custom_records(path: str | Path) -> list[dict]:
    """Read a nonempty record list from CSV, a JSON array, or JSON Lines."""
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".json"}:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            records = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
    else:
        raise ValueError("Custom training data must be .csv, .json, or .jsonl")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Custom training file is empty: {path}")
    return records


def add_custom_training_data(dataset, spec, path: str, target_ratio: float, seed: int = 42):
    """Mix custom examples into train only, oversampling to a requested final ratio."""
    from datasets import Dataset, concatenate_datasets

    if spec.task_type != "single_label":
        raise ValueError("Custom mixed training currently supports single-label datasets")
    if not 0 < target_ratio < 0.5:
        raise ValueError("--extra-ratio must be greater than 0 and less than 0.5")

    records = _read_custom_records(path)
    normalized = []
    seen = set()
    for row_number, record in enumerate(records, start=1):
        if "text" not in record or "label" not in record:
            raise ValueError(f"Custom row {row_number} must contain text and label")
        text = str(record["text"]).strip()
        if not text:
            raise ValueError(f"Custom row {row_number} has empty text")
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"text": text, "labels": parse_external_label(record["label"], spec.label_names)})

    extra = Dataset.from_list(normalized)
    base_count = len(dataset["train"])
    # Solve custom / (base + custom) = target_ratio for the custom row count.
    target_count = max(1, round(base_count * target_ratio / (1.0 - target_ratio)))
    repeat_factor = math.ceil(target_count / len(extra))
    oversampled = concatenate_datasets([extra] * repeat_factor).shuffle(seed=seed).select(range(target_count))
    # Only training is modified; callers must supply custom data free of held-out examples.
    dataset["train"] = concatenate_datasets([dataset["train"], oversampled]).shuffle(seed=seed)
    return dataset, {
        "unique_custom_examples": len(extra),
        "custom_training_rows": target_count,
        "repeat_factor": repeat_factor,
        "final_custom_ratio": target_count / len(dataset["train"]),
    }
