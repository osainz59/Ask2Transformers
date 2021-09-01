import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


def f1_score_(labels, preds, n_labels=42):
    return f1_score(labels, preds, labels=list(range(1, n_labels)), average="micro")


def precision_recall_fscore_(labels, preds, n_labels=42):
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=list(range(1, n_labels)), average="micro")
    return p, r, f


def apply_threshold(output, threshold=0.0, ignore_negative_prediction=True):
    """Applies a threshold to determine whether is a relation or not"""
    output_ = output.copy()
    if ignore_negative_prediction:
        output_[:, 0] = 0.0
    activations = (output_ >= threshold).sum(-1).astype(np.int)
    output_[activations == 0, 0] = 1.00

    return output_.argmax(-1)


def find_optimal_threshold(labels, output, granularity=1000, metric=f1_score_):
    thresholds = np.linspace(0, 1, granularity)
    values = []
    for t in thresholds:
        preds = apply_threshold(output, threshold=t)
        values.append(metric(labels, preds))

    best_metric_id = np.argmax(values)
    best_threshold = thresholds[best_metric_id]

    return best_threshold, values[best_metric_id]
