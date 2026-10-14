from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DATASETS, add_custom_training_data, list_datasets, load_and_prepare


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and use sentiment classifiers")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("datasets", help="List supported public datasets")

    train = commands.add_parser("train", help="Fine-tune a model")
    train.add_argument("--model", choices=("transformer", "lstm"), required=True)
    train.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=float)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--max-train-samples", type=int)
    train.add_argument("--base-model", default="distilbert/distilbert-base-uncased")
    train.add_argument("--max-length", type=int, default=256)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--extra-train",
        help="CSV/JSON/JSONL with text and label fields, mixed into training only",
    )
    train.add_argument(
        "--extra-ratio", type=float, default=0.1,
        help="Target fraction of custom rows after oversampling (default: 0.1)",
    )

    predict = commands.add_parser("predict", help="Classify one sentence")
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--text", required=True)
    predict.add_argument("--threshold", type=float, default=0.5)

    for command, help_text, default_file in (
        (
            "evaluate-sst2", "Run the SST-2 basic and challenge tests",
            Path(__file__).resolve().parents[1] / "examples" / "sst2_test.jsonl",
        ),
        (
            "evaluate-binary", "Evaluate a binary model on any labeled JSONL file",
            None,
        ),
    ):
        evaluate = commands.add_parser(command, help=help_text)
        evaluate.add_argument("--model-dir", required=True)
        evaluate.add_argument(
            "--test-file", required=default_file is None,
            default=str(default_file) if default_file else None,
        )
        evaluate.add_argument("--batch-size", type=int, default=32)
        evaluate.add_argument("--output", help="Optional path for detailed JSON results")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "datasets":
        print(list_datasets())
        return

    if args.command == "predict":
        metadata_path = Path(args.model_dir) / "sentiment_config.json"
        if not metadata_path.exists():
            raise SystemExit(f"Not a trained model directory: {args.model_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["model_type"] == "transformer":
            from .transformer import predict_transformer
            result = predict_transformer(args.model_dir, args.text, args.threshold)
        else:
            from .lstm import predict_lstm
            result = predict_lstm(args.model_dir, args.text, args.threshold)
        print(json.dumps(result, indent=2))
        return

    if args.command in {"evaluate-sst2", "evaluate-binary"}:
        from .evaluate import evaluate_sst2
        result = evaluate_sst2(
            args.model_dir, args.test_file, args.batch_size, args.output
        )
        print(json.dumps(result["summary"], indent=2))
        return

    dataset, spec = load_and_prepare(args.dataset, args.max_train_samples, args.seed)
    if args.extra_train:
        dataset, mix_info = add_custom_training_data(
            dataset, spec, args.extra_train, args.extra_ratio, args.seed
        )
        print("Custom-data mix:", json.dumps(mix_info))
        if mix_info["repeat_factor"] > 10:
            print(
                "WARNING: each custom example is repeated more than 10 times. "
                "Add more unique examples to reduce overfitting."
            )
    if args.model == "transformer":
        from .transformer import train_transformer
        metrics = train_transformer(
            dataset, spec, args.output, base_model=args.base_model,
            epochs=args.epochs or 3.0, batch_size=args.batch_size or 16,
            learning_rate=args.learning_rate or 2e-5, max_length=args.max_length,
            seed=args.seed,
        )
    else:
        from .lstm import train_lstm
        metrics = train_lstm(
            dataset, spec, args.output, epochs=args.epochs or 5.0,
            batch_size=args.batch_size or 64,
            learning_rate=args.learning_rate or 1e-3, max_length=args.max_length,
            seed=args.seed,
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
