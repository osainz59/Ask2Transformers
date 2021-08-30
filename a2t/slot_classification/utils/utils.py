import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

def f1_score_(labels, preds, n_labels=None):
    n_labels = max(labels)+1 if n_labels is None else n_labels
    return f1_score(labels, preds, labels=list(range(1, n_labels)), average='micro')

def individual_f1_score_(labels, preds, n_labels=None):
    n_labels = max(labels)+1 if n_labels is None else n_labels
    return f1_score(labels, preds, labels=list(range(0, n_labels)), average=None)

def precision_recall_fscore_(labels, preds):
    n_labels = max(labels)+1 if n_labels is None else n_labels
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=list(range(1, n_labels)), average='micro')
    return p, r, f

def apply_threshold(output, threshold=0.0, ignore_negative_prediction=True):
    """ Applies a threshold to determine whether is a relation or not
    """
    output_ = output.copy()
    if ignore_negative_prediction:
        output_[:,0] = 0.0
    activations = (output_ >= threshold).sum(-1).astype(np.int)
    output_[activations==0, 0] = 1.00

    return output_.argmax(-1)

def apply_individual_threshold(output, thresholds, ignore_negative_prediction=True):
    output_ = output.copy()
    if ignore_negative_prediction:
        output_[:,0] = 0.0
    for i, threshold in enumerate(thresholds):
        if not i:
            continue
        activations = (output_[:,i] < threshold)
        output_[activations, i] = 0.0
    
    return output_.argmax(-1)
        

def find_optimal_threshold(labels, output, granularity=1000, metric=f1_score_, n_labels=None):
    thresholds = np.linspace(0, 1, granularity)
    values = []
    for t in thresholds:
        preds = apply_threshold(output, threshold=t)
        values.append(metric(labels, preds, n_labels=n_labels))
    
    best_metric_id = np.argmax(values)
    best_threshold = thresholds[best_metric_id]

    return best_threshold, values[best_metric_id]

def find_optimal_individual_threshold(labels, output, granularity=1000, indv_metric=individual_f1_score_, 
                                      metric=f1_score_, n_labels=None, default=.9
    ):
    thresholds = np.linspace(0, 1, granularity)
    values = []
    for t in thresholds:
        preds = apply_individual_threshold(output, thresholds=[t]*output.shape[1])
        values.append(indv_metric(labels, preds, n_labels=n_labels))
    
    best_metric_id = np.argmax(values, 0)
    best_threshold = thresholds[best_metric_id]

    # Fill the thresholds of unseen arguments with default=.5
    if n_labels is not None:
        for i in range(n_labels):
            if i not in np.unique(labels):
                best_threshold[i] = default

    return best_threshold, metric(labels, apply_individual_threshold(output, thresholds=best_threshold))


    