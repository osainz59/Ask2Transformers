from argparse import ArgumentParser
import json
import os
import sys
import numpy as np

sys.path.append('./')
from a2t.relation_classification.tacred import TACRED_LABELS
from a2t.relation_classification.utils import apply_threshold, f1_score_


parser = ArgumentParser()

parser.add_argument('--preds', type=str, default='experiments/re_train_0.01')
parser.add_argument('--data', type=str, default='data/tacred/train.json')
parser.add_argument('--split', type=str, default='data/tacred/splits/0.25.split.txt')
parser.add_argument('--threshold', type=float, default=0.8908)

def main(args):
    with open(args.data) as f:
        train_data = json.load(f)

    preds = np.load(os.path.join(args.preds, 'output.npy'))
    preds = apply_threshold(preds, threshold=args.threshold)
    labels = np.load(os.path.join(args.preds, 'labels.npy'))

    label2id = {v:i for i, v in enumerate(TACRED_LABELS)}

    with open(args.split) as f:
        labeled_split = [line.strip() for line in f]

    new_train_instances = []
    labels_, preds_ = [], []
    for pred, inst in zip(preds, train_data):
        if inst['id'] in labeled_split:
            continue
        # FOR TESTING
        labels_.append(label2id[inst['relation']])
        preds_.append(pred)
        # 
        new_inst = inst.copy()
        new_inst['relation'] = TACRED_LABELS[pred]
        new_train_instances.append(new_inst)

    print(args.preds, f1_score_(labels, preds), f1_score_(labels_, preds_))

    with open(os.path.join(args.preds, 'train.silver.json'), 'wt') as f:
        json.dump(new_train_instances, f, indent=4)

    
 
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)