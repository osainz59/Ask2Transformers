from argparse import ArgumentParser
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict
import numpy as np
import json
import sys
from pprint import pprint
import random

random.seed(0)
np.random.seed(0)

sys.path.append("./")
from a2t.relation_classification.tacred import TACRED_LABEL_TEMPLATES, TACRED_LABELS


@dataclass
class REInputFeatures:
    subj: str
    obj: str
    context: str
    pair_type: str = None
    label: str = None


@dataclass
class MNLIInputFeatures:
    premise: str
    hypothesis: str
    label: int


parser = ArgumentParser()

parser.add_argument("--input_file", type=str, default="data/tacred/dev.json")
# parser.add_argument('--model', type=str, default='microsoft/deberta-v2-xlarge-mnli')
parser.add_argument("--output_file", type=str, default="data/tacred/dev.mnli.json")
parser.add_argument("--negative_pattern", action="store_true", default=False)
parser.add_argument("--negn", type=int, default=1)

args = parser.parse_args()

templates = [
    "{subj} and {obj} are not related",
    "{subj} is also known as {obj}",
    "{subj} was born in {obj}",
    "{subj} is {obj} years old",
    "{obj} is the nationality of {subj}",
    "{subj} died in {obj}",
    "{obj} is the cause of {subj}’s death",
    "{subj} lives in {obj}",
    "{subj} studied in {obj}",
    "{subj} is a {obj}",
    "{subj} is an employee of {obj}",
    "{subj} believe in {obj}",
    "{subj} is the spouse of {obj}",
    "{subj} is the parent of {obj}",
    "{obj} is the parent of {subj}",
    "{subj} and {obj} are siblings",
    "{subj} and {obj} are family",
    "{subj} was convicted of {obj}",
    "{subj} has political affiliation with {obj}",
    "{obj} is a high level member of {subj}",
    "{subj} has about {obj} employees",
    "{obj} is member of {subj}",
    "{subj} is member of {obj}",
    "{obj} is a branch of {subj}",
    "{subj} is a branch of {obj}",
    "{subj} was founded by {obj}",
    "{subj} was founded in {obj}",
    "{subj} existed until {obj}",
    "{subj} has its headquarters in {obj}",
    "{obj} holds shares in {subj}",
    "{obj} is the website of {subj}",
]

labels2id = {"entailment": 2, "neutral": 1, "contradiction": 0}

positive_templates: Dict[str, list] = defaultdict(list)
negative_templates: Dict[str, list] = defaultdict(list)

if not args.negative_pattern:
    templates = templates[1:]

for label in TACRED_LABELS:
    if not args.negative_pattern and label == "no_relation":
        continue
    for template in templates:
        if label != "no_relation" and template == "{subj} and {obj} are not related":
            continue
        if template in TACRED_LABEL_TEMPLATES[label]:
            positive_templates[label].append(template)
        else:
            negative_templates[label].append(template)


def tacred2mnli(
    instance: REInputFeatures,
    positive_templates,
    negative_templates,
    templates,
    negn=1,
    posn=1,
):
    if instance.label == "no_relation":
        template = random.choices(templates, k=negn)
        return [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["contradiction"],
            )
            for t in template
        ]

    # Generate the positive examples
    mnli_instances = []
    positive_template = random.choices(positive_templates[instance.label], k=posn)
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["entailment"],
            )
            for t in positive_template
        ]
    )

    # Generate the negative templates
    negative_template = random.choices(negative_templates[instance.label], k=negn)
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["neutral"],
            )
            for t in negative_template
        ]
    )

    return mnli_instances


def tacred2mnli_with_negative_pattern(
    instance: REInputFeatures,
    positive_templates,
    negative_templates,
    templates,
    negn=1,
    posn=1,
):
    mnli_instances = []
    # Generate the positive examples
    positive_template = random.choices(positive_templates[instance.label], k=posn)
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["entailment"],
            )
            for t in positive_template
        ]
    )

    # Generate the negative templates
    negative_template = random.choices(negative_templates[instance.label], k=negn)
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["neutral"] if instance.label != "no_relation" else labels2id["contradiction"],
            )
            for t in negative_template
        ]
    )

    # Add the contradiction regarding the no_relation pattern if the relation is not no_relation
    if instance.label != "no_relation":
        mnli_instances.append(
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis="{subj} and {obj} are not related.".format(subj=instance.subj, obj=instance.obj),
                label=labels2id["contradiction"],
            )
        )

    return mnli_instances


tacred2mnli = tacred2mnli_with_negative_pattern if args.negative_pattern else tacred2mnli


with open(args.input_file, "rt") as f:
    mnli_data = []
    stats = []
    for line in json.load(f):
        mnli_instance = tacred2mnli(
            REInputFeatures(
                subj=" ".join(line["token"][line["subj_start"] : line["subj_end"] + 1])
                .replace("-LRB-", "(")
                .replace("-RRB-", ")")
                .replace("-LSB-", "[")
                .replace("-RSB-", "]"),
                obj=" ".join(line["token"][line["obj_start"] : line["obj_end"] + 1])
                .replace("-LRB-", "(")
                .replace("-RRB-", ")")
                .replace("-LSB-", "[")
                .replace("-RSB-", "]"),
                pair_type=f"{line['subj_type']}:{line['obj_type']}",
                context=" ".join(line["token"])
                .replace("-LRB-", "(")
                .replace("-RRB-", ")")
                .replace("-LSB-", "[")
                .replace("-RSB-", "]"),
                label=line["relation"],
            ),
            positive_templates,
            negative_templates,
            templates,
            negn=args.negn,
        )
        mnli_data.extend(mnli_instance)
        stats.append(line["relation"] != "no_relation")

with open(args.output_file, "wt") as f:
    for data in mnli_data:
        f.write(f"{json.dumps(data.__dict__)}\n")
    # json.dump([data.__dict__ for data in mnli_data], f, indent=2)

count = Counter([data.label for data in mnli_data])
pprint(count)
count = Counter(stats)
pprint(count)
print(len(stats))
