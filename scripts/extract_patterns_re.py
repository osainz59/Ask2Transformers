from argparse import ArgumentParser
import json

from collections import defaultdict


parser = ArgumentParser()
parser.add_argument('input_file', type=str)
parser.add_argument('output_file', type=str)
parser.add_argument('--max_dist', type=int, default=7)


def main(opt):
    with open(opt.input_file, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    patterns = defaultdict(list)
    for instance in data:
        if min(abs(instance['obj_start'] - instance['subj_end']),  
               abs(instance['obj_end'] - instance['subj_start'])) <= opt.max_dist:

            if instance['subj_start'] < instance['obj_start']:
                pattern = ['{subj}'] + instance['token'][instance['subj_end']+1:instance['obj_start']] + ['{obj}']
            else:
                pattern = ['{obj}'] + instance['token'][instance['obj_end']+1:instance['subj_start']] + ['{subj}']
            patterns[instance['relation']].append(
                " ".join(pattern)
            )

    with open(opt.output_file, 'wt', encoding='utf-8') as f:
        json.dump(patterns, f, indent=4) 

    


if __name__ == "__main__":
    opt = parser.parse_args()
    main(opt)