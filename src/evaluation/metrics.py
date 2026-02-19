import numpy as np
from collections import Counter
from typing import List, Dict


def global_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    return np.mean(np.array(y_true) == np.array(y_pred))


def no_detect_rate(y_pred: List[str], no_label: str = "NO_DETECT") -> float:
    return np.mean(np.array(y_pred) == no_label)


def accuracy_by_class(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    classes = sorted(set(y_true))
    acc = {}
    for c in classes:
        idx = [i for i, t in enumerate(y_true) if t == c]
        if len(idx) == 0:
            continue
        acc[c] = np.mean([y_pred[i] == c for i in idx])
    return acc


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    classes = sorted(set(y_true))
    f1_scores = []

    for c in classes:
        tp = sum((t == c and p == c) for t, p in zip(y_true, y_pred))
        fp = sum((t != c and p == c) for t, p in zip(y_true, y_pred))
        fn = sum((t == c and p != c) for t, p in zip(y_true, y_pred))

        if tp == 0 and (fp == 0 or fn == 0):
            f1_scores.append(0.0)
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return np.mean(f1_scores)


def balanced_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    acc_by = accuracy_by_class(y_true, y_pred)
    return np.mean(list(acc_by.values()))