from pprint import pprint
from typing import Callable, List
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


def f1_score_(labels, preds, n_labels=None):
    n_labels = max(labels) + 1 if n_labels is None else n_labels
    return f1_score(labels, preds, labels=list(range(1, n_labels)), average="micro")


def individual_f1_score_(labels, preds, n_labels=None):
    n_labels = max(labels) + 1 if n_labels is None else n_labels
    return f1_score(labels, preds, labels=list(range(0, n_labels)), average=None)


def precision_recall_fscore_(labels, preds, n_labels: int = None):
    n_labels = max(labels) + 1 if n_labels is None else n_labels
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=list(range(1, n_labels)), average="micro")
    return p, r, f


def apply_threshold(output, threshold=0.0, ignore_negative_prediction=True, negative_label_id: int = 0):
    """Applies a threshold to determine whether is a relation or not"""
    output_ = output.copy()
    if ignore_negative_prediction:
        output_[:, negative_label_id] = 0.0
    activations = (output_ >= threshold).sum(-1).astype(np.int)
    output_[activations == 0, negative_label_id] = 1.00

    return output_.argmax(-1)


def apply_individual_threshold(output, thresholds, ignore_negative_prediction=True):
    output_ = output.copy()
    if ignore_negative_prediction:
        output_[:, 0] = 0.0
    for i, threshold in enumerate(thresholds):
        if not i:
            continue
        activations = output_[:, i] < threshold
        output_[activations, i] = 0.0

    return output_.argmax(-1)


def find_optimal_threshold(
    labels,
    output,
    granularity=1000,
    metric=f1_score_,
    n_labels=None,
    negative_label_id: int = 0,
    apply_threshold_fn: Callable = apply_threshold,
):
    thresholds = np.linspace(0, 1, granularity)
    values = []
    for t in thresholds:
        preds = apply_threshold_fn(output, threshold=t, negative_label_id=negative_label_id)
        values.append(metric(labels, preds, n_labels=n_labels))

    best_metric_id = np.argmax(values)
    best_threshold = thresholds[best_metric_id]

    return best_threshold, values[best_metric_id]


def find_optimal_individual_threshold(
    labels,
    output,
    granularity=1000,
    indv_metric=individual_f1_score_,
    metric=f1_score_,
    n_labels=None,
    default=0.9,
):
    thresholds = np.linspace(0, 1, granularity)
    values = []
    for t in thresholds:
        preds = apply_individual_threshold(output, thresholds=[t] * output.shape[1])
        values.append(indv_metric(labels, preds, n_labels=n_labels))

    best_metric_id = np.argmax(values, 0)
    best_threshold = thresholds[best_metric_id]

    # Fill the thresholds of unseen arguments with default=.5
    if n_labels is not None:
        for i in range(n_labels):
            if i not in np.unique(labels):
                best_threshold[i] = default

    return best_threshold, metric(labels, apply_individual_threshold(output, thresholds=best_threshold))


def apply_multi_label_threshold(output, threshold, **kwargs):
    return [[i for i, coef in enumerate(row) if coef >= threshold] for row in output]


def multi_label_metrics(y_true: List[int], y_pred: List[int], **kwargs):
    tp, total_t, total_p = 0, 0, 0
    for y_t, y_p in zip(y_true, y_pred):
        tp += sum([p in y_t for p in y_p])
        total_t += len(y_t)
        total_p += len(y_p)

    precision = tp / total_p if total_p > 0 else 0.0
    recall = tp / total_t if total_t > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1


if __name__ == "__main__":
    y_true = [[0], [1, 2], [3]]
    output = np.array([[0.9, 0.4, 0.2], [0.7, 0.8, 0.3], [0.2, 0.6, 0.9]])
    threshold, _ = find_optimal_threshold(
        y_true,
        output,
        metric=lambda y_true, y_pred, **kwargs: multi_label_metrics(y_true, y_pred, **kwargs)[-1],
        apply_threshold_fn=apply_multi_label_threshold,
    )
    y_pred = apply_multi_label_threshold(output, threshold)
    pre, rec, f1 = multi_label_metrics(y_true, y_pred)
    pprint(y_pred)
    pprint(threshold)
    print(pre, rec, f1)
