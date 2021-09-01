from argparse import ArgumentParser
import os
import sys
import json

sys.path.append("./")
from a2t.relation_classification.tacred import TACRED_LABEL_TEMPLATES, TACRED_LABELS

from sklearn.model_selection import train_test_split

parser = ArgumentParser()

parser.add_argument("--input_file", type=str, default="data/tacred/train.json")
parser.add_argument("--output_folder", type=str, default="data/tacred/splits")
parser.add_argument(
    "--splits", type=list, default=[0.01, 0.05, 0.1, 0.25, 0.5], nargs="+"
)

args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

label2id = {v: k for k, v, in enumerate(TACRED_LABELS)}

with open(args.input_file, "rt") as f:
    data, ids, labels = {}, [], []
    for line in json.load(f):
        ids.append(line["id"])
        labels.append(label2id[line["relation"]])
        data[line["id"]] = line

prev_split = 0.0
total = 1.0
splits = [[]]
for split in sorted(args.splits):
    test_size = (1.0 / total) * (split - prev_split)
    total -= test_size
    prev_split = split
    ids, test_ids, labels, _ = train_test_split(
        ids, labels, test_size=test_size, stratify=labels
    )
    splits.append(splits[-1] + test_ids)

splits.pop(0)

partition = "train." if "train" in args.input_file else "dev."

for name, split in zip(args.splits, splits):
    os.makedirs(os.path.join(args.output_folder, str(name)), exist_ok=True)
    with open(
        os.path.join(args.output_folder, partition + str(name) + ".split.txt"), "wt"
    ) as f:
        for line in split:
            f.write(line + "\n")
    with open(
        os.path.join(args.output_folder, partition + str(name) + ".json"), "wt"
    ) as f:
        json.dump([data[idx] for idx in split], f)
