from argparse import ArgumentParser
import numpy as np
import glob
import sys
sys.path.append('./')
from a2t.relation_classification.utils import *

parser = ArgumentParser()

parser.add_argument('--name', type=str, default='deberta')

def main(name):
    # Load dev data
    dev_data = {}
    for split in [0.01, 0.05, 0.1]:
        for file_name in glob.iglob(f"experiments/re_dev_{name}_{split}_*"):
            labels = np.load(f"{file_name}/labels.npy")
            output = np.load(f"{file_name}/output.npy")
            dev_data[file_name] = find_optimal_threshold(labels, output)[0]

    test_data = {}
    for split in [0.01, 0.05, 0.1]:
        pre, rec, f1 = [], [], []
        for file_name in glob.iglob(f"experiments/re_test_{name}_{split}_*"):
            labels = np.load(f"{file_name}/labels.npy")
            output = np.load(f"{file_name}/output.npy")
            p, r, f = precision_recall_fscore_(
                labels, apply_threshold(output, dev_data[file_name.replace('test', 'dev')])
            )
            pre.append(p*100)
            rec.append(r*100)
            f1.append(f*100)

        std = np.std(f1)
        output = sorted(list(zip(pre, rec, f1)), key=lambda x: x[-1])[len(f1)//2]
        print(f"{split} - P/R/F1: {output} +/- {std}")

    



if __name__ == "__main__":
    main(parser.parse_args().name)
