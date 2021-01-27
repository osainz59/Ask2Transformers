import gzip
import sys
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('subsample', type=str)
parser.add_argument('glosses', type=str)

args = parser.parse_args()

with open(args.subsample, 'rt', encoding='utf-8') as f:
    synset_list = [line.strip() for line in f]

with gzip.open(args.glosses, 'rt', encoding='utf-8') as f, \
     gzip.open(f"{args.subsample}.gz", 'wt', encoding='utf-8') as wf:
    for line in f:
        if line.strip().split()[0] in synset_list:
            wf.write(line)


