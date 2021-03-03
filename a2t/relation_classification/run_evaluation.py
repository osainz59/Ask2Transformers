import argparse
import json
import os
from pprint import pprint
from collections import Counter

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from .mnli import NLIRelationClassifierWithMappingHead, REInputFeatures
from .tacred import TACRED_LABELS

CLASSIFIERS = {
    'mnli-mapping': NLIRelationClassifierWithMappingHead
}


def top_k_accuracy(output, labels, k=5):
    preds = np.argsort(output)[:, ::-1][:, :k]
    return sum(l in p and l > 0 for l, p in zip(labels, preds)) / (labels > 0).sum()


parser = argparse.ArgumentParser(prog='run_evaluation', description="Run a evaluation for each configuration.")
parser.add_argument('input_file', type=str, default='data/tacred/dev.json',
                    help="Dataset file.")
parser.add_argument('--config', type=str, dest='config',
                    help='Configuration file for the experiment.')

args = parser.parse_args()

labels2id = {label: i for i, label in enumerate(TACRED_LABELS)}

# """
#             "{subj} is also known as {obj}": "org:alternate_names",
#             "{subj} has a headquarter in {obj} city": "org:city_of_headquarters",
#             "{subj} has a headquarter in {obj} country": "org:country_of_headquarters",
#             "{subj} was dissolved by {obj}": "org:dissolved",
#             "{subj} founded {obj}": "org:founded",
#             "{subj} is member of {obj}": "org:member_of",
#             "{subj} is form by {obj}": "org:members",
#             "{subj} has {obj} members": "org:number_of_employees/members",

            # "{subj} has die in {obj} year": "per:date_of_death",
            # "{subj} died in {obj} year": "per:date_of_death",
            # "{subj} has passed away in {obj} year": "per:date_of_death"


            #             "{obj} was where {subj} died": "per:city_of_death",
            # "{subj} has die in {obj} city": "per:city_of_death",
            # "{subj} died in {obj} city": "per:city_of_death",
            # "{subj} has passed away in {obj} city": "per:city_of_death",
# """

with open(args.input_file, 'rt') as f:
    features, labels = [], []
    for line in json.load(f):
        if line['relation'] not in [
            "org:alternate_names",
            "org:parents",
            # "org:city_of_headquarters",
            # "org:country_of_headquarters",
            "org:dissolved",
            "org:founded",
            "org:founded_by",
            "org:member_of",
            "org:members",
            "org:number_of_employees/members",
            "per:schools_attended",
            "per:siblings",
            "per:religion",
            "per:date_of_death",
            "per:city_of_death",
            "per:country_of_death",
            "per:spouse",
            "per:parents",
            "per:title",
            "no_relation"]:
            continue
        features.append(REInputFeatures(
            subj=" ".join(line['token'][line['subj_start']:line['subj_end']+1]).replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', ']'),
            obj=" ".join(line['token'][line['obj_start']:line['obj_end']+1]).replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', ']'),
            context= " ".join(line['token']).replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', ']'),
            label=line['relation']
        ))
        labels.append(labels2id[line['relation']])
        # if line['relation'] == 'per:parents':
        #     pprint(features[-1])
        #     print()

labels = np.array(labels)

with open(args.config, 'rt') as f:
    config = json.load(f)

for configuration in config:
    n_labels = len(TACRED_LABELS)
    os.makedirs(f"experiments/{configuration['name']}", exist_ok=True)
    classifier = CLASSIFIERS[configuration['classification_model']](**configuration)
    output = classifier(features, batch_size=configuration['batch_size'], multiclass=configuration['multiclass'])
    np.save(f"experiments/{configuration['name']}/output.npy", output)
    np.save(f"experiments/{configuration['name']}/labels.npy", labels)
    pre, rec, f1, _ = precision_recall_fscore_support(labels, np.argmax(output, -1), average='micro', labels=list(range(1, n_labels)))
    cm = confusion_matrix(labels, np.argmax(output, -1))
    configuration['precision'] = pre
    configuration['recall'] = rec
    configuration['f1-score'] = f1
    configuration['top-1'] = top_k_accuracy(output, labels, k=1)
    configuration['top-3'] = top_k_accuracy(output, labels, k=3)
    configuration['top-5'] = top_k_accuracy(output, labels, k=5)
    configuration['topk-curve'] = [top_k_accuracy(output, labels, k=i) for i in range(1, n_labels+1)]
    configuration['confusion_matrix'] = cm.tolist()
    configuration['indices'] = { TACRED_LABELS[key]: int(key) for key, value in Counter(labels).items()}
    pprint(configuration)

with open(args.config, 'wt') as f:
    json.dump(config, f, indent=4)
