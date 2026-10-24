"""NumPy metrics for mutually exclusive classes and independent labels."""

from __future__ import annotations

import numpy as np


def single_label_metrics(logits, labels) -> dict[str, float]:
    """Score [examples, classes] logits against integer class IDs.

    Macro F1 weights each class present in targets or predictions equally.
    """
    predictions = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(labels)
    accuracy = float((predictions == labels).mean())
    f1_values = []
    for class_id in sorted(set(labels.tolist()) | set(predictions.tolist())):
        tp = int(((predictions == class_id) & (labels == class_id)).sum())
        fp = int(((predictions == class_id) & (labels != class_id)).sum())
        fn = int(((predictions != class_id) & (labels == class_id)).sum())
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return {"accuracy": accuracy, "macro_f1": float(np.mean(f1_values))}


def multi_label_metrics(logits, labels, threshold: float = 0.5) -> dict[str, float]:
    """Score independent label logits against multi-hot targets using sigmoid.

    Exact match requires every label in an example to agree; micro F1 pools
    true positives, false positives, and false negatives across all labels.
    """
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    predictions = probabilities >= threshold
    labels = np.asarray(labels) >= 0.5
    exact_match = float(np.all(predictions == labels, axis=1).mean())
    tp = int((predictions & labels).sum())
    fp = int((predictions & ~labels).sum())
    fn = int((~predictions & labels).sum())
    denominator = 2 * tp + fp + fn
    micro_f1 = 0.0 if denominator == 0 else 2 * tp / denominator
    return {"exact_match": exact_match, "micro_f1": float(micro_f1)}

