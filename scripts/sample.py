import gzip
import sys
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('subsample', type=str)
parser.add_argument('glosses', type=str)
parser.add_argument('--labels', type=str, default=None)
parser.add_argument('--label_type', type=str, default='babel')

args = parser.parse_args()

if args.labels:
    label_type = {
        'babel': 1,
        'wndomains': 2
    }.get(args.label_type, 1)
    with open(args.labels, 'rt') as f:
        labels = {}
        for line in f:
            row = line.strip().split('\t')
            idx, label = row[0], row[label_type]
            labels[idx] = label if label != 'None' else 'factotum'

with open(args.subsample, 'rt', encoding='utf-8') as f:
    synset_list = [line.strip() for line in f]

with gzip.open(args.glosses, 'rt', encoding='utf-8') as f, \
     gzip.open(f"{args.subsample}.gz", 'wt', encoding='utf-8') as wf:
    for line in f:
        if not args.labels:
            if line.strip().split()[0] in synset_list:
                wf.write(line)
        else:
            row = line.split()
            idx, gloss = row[0], " ".join(row[1:])
            gloss = gloss.replace('\t', ' ')
            if line.strip().split()[0] in synset_list:
                try:
                    print(f"{idx}\t{labels.get(idx, 'NULL')}\t{gloss}")
                except:
                    continue


