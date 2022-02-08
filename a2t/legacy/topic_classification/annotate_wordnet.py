import argparse
import gzip
import json
import os

import numpy as np

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

parser = argparse.ArgumentParser(
    prog="annotate_wordnet",
    description="Annotate WordNet glosses with domain information.",
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

senses, glosses = [], []
with gzip.open(args.dataset, "rt") as f:
    for line in f:
        row = line.split()
        idx, gloss = row[0], " ".join(row[1:])
        gloss = gloss.replace("\t", " ")
        senses.append(idx)
        glosses.append(gloss)

with open(args.config, "rt") as f:
    config = json.load(f)

for configuration in config:
    os.makedirs(f"experiments/{configuration['name']}", exist_ok=True)
    classifier = CLASSIFIERS[configuration["classification_model"]](labels=topics, **configuration)
    output = classifier(glosses, batch_size=configuration["batch_size"])
    # Normalize the output to avoid low confidence level due to large ammount of labels
    output = (output - output.min()) / (output.max() - output.min())
    labels = [topics[i] for i in np.argmax(output, -1)]
    confidences = [x for x in np.max(output, -1)]

    output_name = os.path.basename(args.dataset).replace("list", "annotations")

    with gzip.open(f"experiments/{configuration['name']}/{output_name}", "wt") as out_f:
        for sense, label, confidence in zip(senses, labels, confidences):
            out_f.write(f"{sense}\t{label}\t{confidence}\n")
