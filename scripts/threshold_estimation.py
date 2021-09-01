import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import json

import sys, os
from pprint import pprint
import random

random.seed(0)
np.random.seed(0)

sys.path.append("./")
from a2t.relation_classification.utils import *


parser = argparse.ArgumentParser()

parser.add_argument("--dev", type=str, default="experiments/re_dev")
parser.add_argument("--test", type=str, default="experiments/re_test")
parser.add_argument("--splits", type=int, default=[1000, 100, 10, 4, 2], nargs="+")
parser.add_argument("--output", type=str, default=".ignore/threshold_estimation")


def main(args):
    # Read dev and test outputs
    dev_labels = np.load(os.path.join(args.dev, "labels.npy"))
    dev_output = np.load(os.path.join(args.dev, "output.npy"))

    test_labels = np.load(os.path.join(args.test, "labels.npy"))
    test_output = np.load(os.path.join(args.test, "output.npy"))

    # Compute the best threshold
    best_threshold, best_f1 = find_optimal_threshold(test_labels, test_output)

    # Define the result dict
    result_dict = defaultdict(list)
    result_dict["best"].append([best_threshold, best_f1, 0.0, 0.0])

    threshold, _ = find_optimal_threshold(dev_labels, dev_output)
    p, r, f1 = precision_recall_fscore_(
        test_labels, apply_threshold(test_output, threshold=threshold)
    )
    result_dict[1].append(
        [threshold, f1, abs(best_threshold - threshold), abs(best_f1 - f1), p, r]
    )

    # Estimate the threshold for each split
    for n_splits in args.splits:
        kfold = StratifiedKFold(n_splits=n_splits)
        for _, idx in kfold.split(dev_output, dev_labels):
            splitted_output, splitted_labels = dev_output[idx], dev_labels[idx]
            threshold, _ = find_optimal_threshold(splitted_labels, splitted_output)
            p, r, f1 = precision_recall_fscore_(
                test_labels, apply_threshold(test_output, threshold=threshold)
            )
            result_dict[n_splits].append(
                [
                    threshold,
                    f1,
                    abs(best_threshold - threshold),
                    abs(best_f1 - f1),
                    p,
                    r,
                ]
            )

    print(f"Dev %\tMAbsE\tMean F1")
    for key in sorted([key for key in result_dict.keys() if key != "best"]):
        mean_error = np.mean([x[2] for x in result_dict[key]])
        mean_f1 = np.mean([x[1] for x in result_dict[key]])
        name = f"{100/key:.1f}" if isinstance(key, int) else key
        print(f"{name}\t{mean_error:.4f}\t{mean_f1*100:.2f}")
    # pprint(result_dict)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "result_dict.json"), "wt") as f:
            json.dump(result_dict, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
