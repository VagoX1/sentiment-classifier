# Sentiment Classifier

Train, evaluate, and compare Transformer and bidirectional LSTM models for text
sentiment classification. The project supports public Hugging Face datasets,
mixed-domain fine-tuning with app reviews, single-sentence prediction, and
diagnostic tests for difficult language such as sarcasm and negation.

## Highlights

- Fine-tune DistilBERT, RoBERTa, DeBERTa, or another compatible Transformer.
- Train a bidirectional LSTM baseline with PyTorch.
- Handle binary, multiclass, and multilabel classification.
- Mix custom CSV, JSON, or JSONL examples into a public training dataset.
- Convert real app reviews into balanced, mutually disjoint splits.
- Evaluate ordinary accuracy and linguistic robustness separately.
- Save models and run predictions through one command-line interface.

## Current results

The current RoBERTa experiment was trained on SST-2 plus app-review examples.

| Evaluation | Result |
|---|---:|
| App-review test accuracy | 89.4% (1,788/2,000) |
| Negative-class recall | 91.1% |
| Positive-class recall | 87.7% |
| Original SST-2 model validation accuracy | 90.6% |
| Original SST-2 model challenge accuracy | 75.0% (36/48) |

The app test labels are derived from star ratings, so the results measure
agreement with those ratings. Some source reviews are noisy, unclear, or written
in languages other than English despite the dataset's English designation.

## Project structure

```text
sentiment-classifier/
├── sentiment_classifier/       Training, prediction, and evaluation code
├── examples/                   Hand-authored prompts and challenge examples
├── tests/                      Offline unit tests
├── prepare_app_reviews.py      App-review dataset converter
├── requirements.txt
└── README.md
```

Generated datasets are written under `data/`, while trained models and reports
are written under `outputs/`. Both directories are excluded from Git.

## Installation

Python 3.10 or newer is recommended.

```powershell
git clone https://github.com/YOUR_USERNAME/sentiment-classifier.git
cd sentiment-classifier

python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The execution-policy command is only needed if PowerShell blocks virtual
environment activation.

## Quick start

List the supported public datasets:

```powershell
python -m sentiment_classifier datasets
```

Train a small Transformer smoke test:

```powershell
python -m sentiment_classifier train `
  --model transformer `
  --dataset sst2 `
  --output outputs\sst2-transformer-smoke `
  --max-train-samples 2000 `
  --epochs 1
```

Classify a sentence:

```powershell
python -m sentiment_classifier predict `
  --model-dir outputs\sst2-transformer-smoke `
  --text "The new version is surprisingly good."
```

The prediction command automatically detects whether the saved model is a
Transformer or an LSTM.

## Supported datasets

| CLI name | Task | Domain |
|---|---|---|
| `sst2` | Positive/negative | Movie-review sentences |
| `imdb` | Positive/negative | Full movie reviews |
| `tweet_eval` | Negative/neutral/positive | Social-media posts |
| `amazon_polarity` | Positive/negative | Product reviews |
| `yelp` | One to five stars | Business reviews |
| `financial_phrasebank` | Negative/neutral/positive | Financial news |
| `go_emotions` | 28-label multilabel classification | Reddit emotions |

Datasets and models are downloaded from Hugging Face when a command first uses
them. Always inspect the relevant dataset card and license before commercial
use. Financial PhraseBank, for example, carries non-commercial restrictions.

## Model training

### Transformer

```powershell
python -m sentiment_classifier train `
  --model transformer `
  --dataset sst2 `
  --base-model FacebookAI/roberta-base `
  --output outputs\sst2-roberta `
  --epochs 3
```

The default base model is `distilbert/distilbert-base-uncased`. Pass another
Hugging Face sequence-classification-compatible checkpoint with `--base-model`.

### Bidirectional LSTM

```powershell
python -m sentiment_classifier train `
  --model lstm `
  --dataset sst2 `
  --output outputs\sst2-lstm `
  --epochs 5
```

The LSTM is useful as a lightweight baseline. A pretrained Transformer will
usually provide better accuracy and robustness.

## Mixed-domain fine-tuning

SST-2 teaches general sentiment using movie-review language. For app sentiment,
mix it with real app reviews and targeted contrastive examples.

Custom data must be CSV, JSON, or JSONL and contain `text` and `label` fields:

```json
{"text": "Wonderful, the app deleted all my work again.", "label": 0}
{"text": "Wonderful, the app recovered all my work.", "label": 1}
```

Binary labels can be `0`/`negative` or `1`/`positive`.

The included
[`examples/app_sarcasm_train.jsonl`](examples/app_sarcasm_train.jsonl) contains
small contrastive pairs for pipeline experiments. It is not large enough to be
a production training corpus.

### Prepare real app reviews

The converter downloads the MIT-licensed
`Sharathhebbar24/app_reviews_modded` dataset and applies the following mapping:

- One or two stars → negative
- Four or five stars → positive
- Three stars → excluded

It cleans HTML and whitespace, removes duplicate text, balances both classes,
and prevents overlap across the generated splits.

```powershell
python prepare_app_reviews.py `
  --train-per-label 5000 `
  --validation-per-label 1000 `
  --test-per-label 1000
```

Generated files:

| File | Rows | Purpose |
|---|---:|---|
| `data/app_reviews_train.jsonl` | 10,000 | Fine-tuning |
| `data/app_reviews_validation.jsonl` | 2,000 | Model selection |
| `data/app_reviews_test.jsonl` | 2,000 | Final evaluation only |

### Train the mixed model

```powershell
python -m sentiment_classifier train `
  --model transformer `
  --dataset sst2 `
  --extra-train data\app_reviews_train.jsonl `
  --extra-ratio 0.15 `
  --base-model FacebookAI/roberta-base `
  --output outputs\sst2-app-roberta `
  --epochs 3
```

`--extra-ratio 0.15` makes custom examples approximately 15% of the combined
training rows. When a custom file is small, the CLI oversamples it and warns if
the repetition factor is high.

## Evaluation

### App-review test set

Use validation results while choosing models and hyperparameters. Run the test
set only for the final measurement:

```powershell
python -m sentiment_classifier evaluate-binary `
  --model-dir outputs\sst2-app-roberta `
  --test-file data\app_reviews_test.jsonl `
  --output outputs\sst2-app-roberta\app-test-results.json
```

### Linguistic challenge suite

[`examples/sst2_test.jsonl`](examples/sst2_test.jsonl) contains 12 basic and 48
challenge examples covering:

- Negation and quantifier scope
- Contrast and comparisons
- Implicit sentiment
- Sarcasm
- Quotation
- Idioms
- Long-distance composition

```powershell
python -m sentiment_classifier evaluate-sst2 `
  --model-dir outputs\sst2-app-roberta `
  --output outputs\sst2-app-roberta\challenge-results.json
```

The evaluator reports overall, tier-level, and per-category accuracy. Never add
these challenge examples to training; doing so would contaminate the benchmark.

## Manual test prompts

[`examples/prompts.json`](examples/prompts.json) contains labeled examples from
general sentiment, product and movie reviews, social media, finance, negation,
sarcasm, mixed sentiment, and emotion classification.

## Tests

The core test suite does not download datasets or model checkpoints:

```powershell
python -m unittest discover -s tests -v
```

## Limitations

- Sarcasm often requires context beyond a single sentence.
- Star ratings can disagree with the written review.
- Dataset language, domain, and demographic biases transfer to the model.
- A high benchmark score does not guarantee reliable production behavior.
- The classifier should not be used as the sole basis for high-impact decisions.

## Model files

Large checkpoints under `outputs/` are intentionally excluded from GitHub.
Publish trained weights to the Hugging Face Hub or store them with another
large-file service. Never commit access tokens, credentials, or virtual
environments.
