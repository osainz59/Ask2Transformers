import argparse
import json
import os
from pprint import pprint
from collections import Counter
import torch

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from .mnli import NLIRelationClassifierWithMappingHead, REInputFeatures
from .tacred import *
from .utils import find_optimal_threshold, apply_threshold

CLASSIFIERS = {"mnli-mapping": NLIRelationClassifierWithMappingHead}


def top_k_accuracy(output, labels, k=5):
    preds = np.argsort(output)[:, ::-1][:, :k]
    return sum(l in p and l > 0 for l, p in zip(labels, preds)) / (labels > 0).sum()


parser = argparse.ArgumentParser(
    prog="run_evaluation",
    description="Run a evaluation for each configuration.",
)
parser.add_argument(
    "input_file",
    type=str,
    default="data/tacred/dev.json",
    help="Dataset file.",
)
parser.add_argument(
    "--config",
    type=str,
    dest="config",
    help="Configuration file for the experiment.",
)
parser.add_argument("--basic", action="store_true", default=False)

args = parser.parse_args()

labels2id = (
    {label: i for i, label in enumerate(TACRED_LABELS)}
    if not args.basic
    else {label: i for i, label in enumerate(TACRED_BASIC_LABELS)}
)

with open(args.input_file, "rt") as f:
    features, labels = [], []
    for i, line in enumerate(json.load(f)):
        line["relation"] = (
            line["relation"] if not args.basic else TACRED_BASIC_LABELS_MAPPING.get(line["relation"], line["relation"])
        )
        features.append(
            REInputFeatures(
                subj=" ".join(line["token"][line["subj_start"] : line["subj_end"] + 1])
                .replace("-LRB-", "(")
                .replace("-RRB-", ")")
                .replace("-LSB-", "[")
                .replace("-RSB-", "]"),
                obj=" ".join(line["token"][line["obj_start"] : line["obj_end"] + 1])
                .replace("-LRB-", "(")
                .replace("-RRB-", ")")
                .replace("-LSB-", "[")
                .replace("-RSB-", "]"),
                pair_type=f"{line['subj_type']}:{line['obj_type']}",
                context=" ".join(line["token"])
                .replace("-LRB-", "(")
                .replace("-RRB-", ")")
                .replace("-LSB-", "[")
                .replace("-RSB-", "]"),
                label=line["relation"],
            )
        )
        labels.append(labels2id[line["relation"]])

labels = np.array(labels)

with open(args.config, "rt") as f:
    config = json.load(f)

LABEL_LIST = TACRED_BASIC_LABELS if args.basic else TACRED_LABELS

for configuration in config:
    n_labels = len(LABEL_LIST)
    os.makedirs(f"experiments/{configuration['name']}", exist_ok=True)
    _ = configuration.pop("negative_threshold", None)
    classifier = CLASSIFIERS[configuration["classification_model"]](negative_threshold=0.0, **configuration)
    output = classifier(
        features,
        batch_size=configuration["batch_size"],
        multiclass=configuration["multiclass"],
    )
    if not "use_threshold" in configuration or configuration["use_threshold"]:
        optimal_threshold, _ = find_optimal_threshold(labels, output)
        output_ = apply_threshold(output, threshold=optimal_threshold)
    else:
        output_ = output.argmax(-1)
    pre, rec, f1, _ = precision_recall_fscore_support(labels, output_, average="micro", labels=list(range(1, n_labels)))

    np.save(f"experiments/{configuration['name']}/output.npy", output)
    np.save(f"experiments/{configuration['name']}/labels.npy", labels)
    configuration["precision"] = pre
    configuration["recall"] = rec
    configuration["f1-score"] = f1
    configuration["top-1"] = top_k_accuracy(output, labels, k=1)
    configuration["top-3"] = top_k_accuracy(output, labels, k=3)
    configuration["top-5"] = top_k_accuracy(output, labels, k=5)
    configuration["topk-curve"] = [top_k_accuracy(output, labels, k=i) for i in range(1, n_labels + 1)]
    _ = configuration.pop("topk-curve", None)
    configuration["negative_threshold"] = optimal_threshold

    pprint(configuration)
    del classifier
    torch.cuda.empty_cache()


with open(args.config, "wt") as f:
    json.dump(config, f, indent=4)
