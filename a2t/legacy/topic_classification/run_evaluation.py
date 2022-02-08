import argparse
import json
import os
from pprint import pprint

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from a2t.legacy.topic_classification.mlm import MLMTopicClassifier
from a2t.legacy.topic_classification.mnli import (
    NLITopicClassifier,
    NLITopicClassifierWithMappingHead,
)
from a2t.legacy.topic_classification.nsp import NSPTopicClassifier

CLASSIFIERS = {
    "mnli": NLITopicClassifier,
    "nsp": NSPTopicClassifier,
    "mlm": MLMTopicClassifier,
    "mnli-mapping": NLITopicClassifierWithMappingHead,
}


def top_k_accuracy(output, labels, k=5):
    preds = np.argsort(output)[:, ::-1][:, :k]
    return sum(l in p for l, p in zip(labels, preds)) / len(labels)


parser = argparse.ArgumentParser(
    prog="run_evaluation",
    description="Run a evaluation for each configuration.",
)
parser.add_argument("dataset", type=str, help="Dataset file.")
parser.add_argument("topics", type=str, help="Topics or classes file.")
parser.add_argument(
    "--config",
    type=str,
    dest="config",
    help="Configuration file for the experiment.",
)

args = parser.parse_args()

with open(args.topics, "rt") as f:
    topics = [topic.rstrip().replace("_", " ") for topic in f]

topic2id = {topic: i for i, topic in enumerate(topics)}

with open(args.dataset, "rt") as f:
    contexts, labels = [], []
    for line in f:
        _, label, context = line.strip().split("\t")
        contexts.append(context)
        labels.append(topic2id[label])

labels = np.array(labels)

with open(args.config, "rt") as f:
    config = json.load(f)

for configuration in config:
    os.makedirs(f"experiments/{configuration['name']}", exist_ok=True)
    classifier = CLASSIFIERS[configuration["classification_model"]](labels=topics, **configuration)
    output = classifier(contexts, batch_size=configuration["batch_size"])
    np.save(f"experiments/{configuration['name']}/output.npy", output)
    np.save(f"experiments/{configuration['name']}/labels.npy", labels)
    pre, rec, f1, _ = precision_recall_fscore_support(labels, np.argmax(output, -1), average="weighted")
    configuration["precision"] = pre
    configuration["recall"] = rec
    configuration["f1-score"] = f1
    configuration["top-1"] = top_k_accuracy(output, labels, k=1)
    configuration["top-3"] = top_k_accuracy(output, labels, k=3)
    configuration["top-5"] = top_k_accuracy(output, labels, k=5)
    configuration["topk-curve"] = [top_k_accuracy(output, labels, k=i) for i in range(len(topics))]
    pprint(configuration)
    print(pre, rec, f1, top_k_accuracy(output, labels, k=1))

with open(args.config, "wt") as f:
    json.dump(config, f, indent=4)
