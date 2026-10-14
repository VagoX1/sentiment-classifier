import unittest
from pathlib import Path

import numpy as np
import torch

from sentiment_classifier.data import DATASETS, parse_external_label
from sentiment_classifier.lstm import SentimentLSTM, build_vocabulary, tokenize_text
from sentiment_classifier.metrics import multi_label_metrics, single_label_metrics
from sentiment_classifier.evaluate import read_jsonl
from prepare_app_reviews import clean_review, rating_from_feature, sentiment_from_rating


class CoreTests(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(set(DATASETS), {
            "sst2", "imdb", "tweet_eval", "amazon_polarity", "yelp",
            "financial_phrasebank", "go_emotions",
        })

    def test_metrics(self):
        result = single_label_metrics([[5, 1], [1, 5]], [0, 1])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)
        multi = multi_label_metrics([[10, -10], [-10, 10]], [[1, 0], [0, 1]])
        self.assertEqual(multi["micro_f1"], 1.0)

    def test_lstm_forward(self):
        model = SentimentLSTM(vocabulary_size=10, num_labels=3, embedding_size=8, hidden_size=4)
        output = model(torch.tensor([[2, 3, 0], [4, 5, 6]]), torch.tensor([2, 3]))
        self.assertEqual(tuple(output.shape), (2, 3))

    def test_tokenizer_and_vocabulary(self):
        self.assertEqual(tokenize_text("It's good!"), ["it's", "good", "!"])
        vocabulary = build_vocabulary(["good good", "bad"], min_frequency=2)
        self.assertIn("good", vocabulary)
        self.assertNotIn("bad", vocabulary)

    def test_sst2_challenge_set(self):
        path = Path(__file__).parents[1] / "examples" / "sst2_test.jsonl"
        records = read_jsonl(path)
        self.assertGreaterEqual(len(records), 50)
        self.assertEqual({record["label"] for record in records}, {0, 1})
        self.assertIn("challenge", {record["split"] for record in records})
        self.assertGreaterEqual(len({record["category"] for record in records}), 8)

    def test_custom_labels(self):
        names = ("negative", "positive")
        self.assertEqual(parse_external_label("negative", names), 0)
        self.assertEqual(parse_external_label("1", names), 1)
        with self.assertRaises(ValueError):
            parse_external_label("neutral", names)

    def test_app_review_conversion(self):
        self.assertEqual(clean_review(" Great<br /> app &amp; fast!  "), "Great app & fast!")
        self.assertEqual(rating_from_feature("5"), 5)
        self.assertEqual(sentiment_from_rating(1), 0)
        self.assertIsNone(sentiment_from_rating(3))
        self.assertEqual(sentiment_from_rating(5), 1)


if __name__ == "__main__":
    unittest.main()
