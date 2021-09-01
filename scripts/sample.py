import gzip
import sys
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--subsample", type=str, default="data/subsample.synset.list")
parser.add_argument("--glosses", type=str, default="data/all.definitions.gz")
parser.add_argument(
    "--labels",
    type=str,
    nargs="+",
    default=[
        "experiments/splitted_topics/all.definitions.gz",
        "experiments/splitted_topics_single/all.definitions.gz",
        "experiments/wndomains/all.definitions.gz",
        "data/topic_classification/gold/babeldomains.silver.txt",
        "data/topic_classification/gold/wndomains.txt",
    ],
)
parser.add_argument("--output_file", type=str, default=".ignore/subsample.annotations.tsv.gz")

args = parser.parse_args()

with open(args.subsample, "rt", encoding="utf-8") as f:
    synset_list = [line.strip() for line in f]


def normalize_idx(idx):
    if idx.startswith("n") or idx.startswith("a") or idx.startswith("v") or idx.startswith("r"):
        idx = idx[1:] + "-" + idx[0]

    return idx


all_labels = []
for label_file in args.labels:
    open_file = open if ".gz" not in label_file else gzip.open
    labels_dict = {}
    with open_file(label_file, "rt") as f:
        for line in f:
            line = line.strip().split("\t")
            idx = normalize_idx(line[0])
            label = "\t".join(line[1:3]) if len(line) > 2 else "\t".join(line[1:3]) + "\t1.0"
            labels_dict[idx] = label
    all_labels.append(labels_dict)

with gzip.open(args.glosses, "rt", encoding="utf-8") as f, gzip.open(args.output_file, "wt", encoding="utf-8") as wf:

    wf.write(
        f"synset-id\tA2T_babeldomains\tA2T_babeldomains_score\tA2T_babeldomains_simple\tA2T_babeldomains_simple_score\tA2T_wndomains\tA2T_wndomains_score\tbabeldomains\tbabeldomains_score\tWNDomains\tWNDomains_score\tgloss\n"
    )
    for line in f:
        row = line.split()
        idx, gloss = row[0], " ".join(row[1:])
        if idx not in synset_list:
            continue
        output_row = [idx] + [labels_dict.get(idx, "NULL\t0.0") for labels_dict in all_labels] + [gloss]
        output_row = "\t".join(output_row)
        wf.write(f"{output_row}\n")
