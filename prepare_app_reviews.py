"""Convert Hugging Face app reviews into balanced binary sentiment JSONL files."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


WHITESPACE = re.compile(r"\s+")


def clean_review(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = text.replace("<br />", " ").replace("<br>", " ")
    return WHITESPACE.sub(" ", text).strip()


def rating_from_feature(value: object, feature=None) -> int:
    """Decode a datasets.ClassLabel star ID, falling back to a literal rating."""
    if feature is not None and hasattr(feature, "int2str"):
        return int(feature.int2str(int(value)))
    rating = int(value)
    if not 1 <= rating <= 5:
        raise ValueError(f"Expected a star rating from 1 to 5, got {value!r}")
    return rating


def sentiment_from_rating(rating: int) -> int | None:
    if rating in (1, 2):
        return 0
    if rating in (4, 5):
        return 1
    if rating == 3:
        return None
    raise ValueError(f"Expected a star rating from 1 to 5, got {rating}")


def collect_balanced(
    split,
    star_feature,
    per_label: int,
    seed: int,
    min_chars: int,
    max_chars: int,
    excluded_texts: set[str] | None = None,
) -> tuple[list[dict], set[str]]:
    excluded_texts = excluded_texts or set()
    seen = set(excluded_texts)
    records: dict[int, list[dict]] = {0: [], 1: []}

    # Convert the two required columns once, then clean only a randomized pool
    # near the requested size. This is substantially faster than decoding every
    # Arrow row individually on datasets with hundreds of thousands of reviews.
    frame = split.select_columns(["review", "star"]).to_pandas()
    rating_lookup = {
        value: rating_from_feature(value, star_feature)
        for value in frame["star"].unique().tolist()
    }
    frame["rating"] = frame["star"].map(rating_lookup)

    for label, accepted_ratings in ((0, (1, 2)), (1, (4, 5))):
        candidates = frame[frame["rating"].isin(accepted_ratings)]
        pool_size = min(len(candidates), max(per_label * 3, per_label + 1000))
        candidates = candidates.sample(n=pool_size, random_state=seed + label)
        for row in candidates.itertuples(index=False):
            text = clean_review(row.review)
            key = text.casefold()
            if not min_chars <= len(text) <= max_chars or key in seen:
                continue
            seen.add(key)
            records[label].append({"text": text, "label": label, "rating": int(row.rating)})
            if len(records[label]) >= per_label:
                break

    available = min(len(records[0]), len(records[1]))
    if available == 0:
        raise RuntimeError("No usable positive/negative examples were found")
    if available < per_label:
        print(f"WARNING: requested {per_label} per label but only found {available}")
    balanced = records[0][:available] + records[1][:available]

    # A deterministic final shuffle avoids writing all negative rows first.
    import random

    random.Random(seed).shuffle(balanced)
    return balanced, seen


def write_jsonl(records: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare balanced binary sentiment data from English app reviews"
    )
    parser.add_argument(
        "--dataset", default="Sharathhebbar24/app_reviews_modded",
        help="Hugging Face dataset repository",
    )
    parser.add_argument("--train-output", default="data/app_reviews_train.jsonl")
    parser.add_argument("--validation-output", default="data/app_reviews_validation.jsonl")
    parser.add_argument("--test-output", default="data/app_reviews_test.jsonl")
    parser.add_argument("--train-per-label", type=int, default=5000)
    parser.add_argument("--validation-per-label", type=int, default=1000)
    parser.add_argument("--test-per-label", type=int, default=1000)
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.train_per_label < 1 or args.validation_per_label < 1 or args.test_per_label < 1:
        raise SystemExit("Sample counts must be positive")
    if args.min_chars < 1 or args.max_chars < args.min_chars:
        raise SystemExit("Invalid character-length limits")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install dependencies with: python -m pip install -r requirements.txt") from exc

    print(f"Loading {args.dataset} ...")
    dataset = load_dataset(args.dataset)
    if "train" not in dataset:
        raise SystemExit("Dataset does not contain a train split")
    if "test" not in dataset:
        raise SystemExit("Dataset does not contain a test split")

    star_feature = dataset["train"].features.get("star")
    train_records, train_texts = collect_balanced(
        dataset["train"], star_feature, args.train_per_label, args.seed,
        args.min_chars, args.max_chars,
    )
    validation_records, train_and_validation_texts = collect_balanced(
        dataset["train"], star_feature, args.validation_per_label, args.seed + 1,
        args.min_chars, args.max_chars, excluded_texts=train_texts,
    )
    test_records, _ = collect_balanced(
        dataset["test"], star_feature, args.test_per_label, args.seed + 2,
        args.min_chars, args.max_chars, excluded_texts=train_and_validation_texts,
    )
    write_jsonl(train_records, args.train_output)
    write_jsonl(validation_records, args.validation_output)
    write_jsonl(test_records, args.test_output)
    print(f"Wrote {len(train_records):,} training examples to {args.train_output}")
    print(f"Wrote {len(validation_records):,} validation examples to {args.validation_output}")
    print(f"Wrote {len(test_records):,} test examples to {args.test_output}")
    print("Mapping: 1–2 stars = negative, 4–5 stars = positive, 3 stars = excluded")


if __name__ == "__main__":
    main()
