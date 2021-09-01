from argparse import ArgumentParser
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, Any
import numpy as np
import json
import sys
from pprint import pprint
import random

random.seed(0)
np.random.seed(0)

sys.path.append("./")
from a2t.slot_classification.wikievents import WikiEventsArgumentDataset
from a2t.slot_classification.data import SlotFeatures

from transformers import AutoConfig


@dataclass
class MNLIInputFeatures:
    premise: str
    hypothesis: str
    label: int


parser = ArgumentParser()

parser.add_argument("--input_file", type=str, default="data/wikievents/dev.jsonl")
parser.add_argument(
    "--config_file", type=str, default="experiments/slot_classification/config.json"
)
parser.add_argument("--output_file", type=str, default="data/wikievents/dev.mnli.jsonl")
parser.add_argument("--model_name_or_path", type=str, default=None)
parser.add_argument("--negn", type=int, default=1)
parser.add_argument("--posn", type=int, default=1)


def generate_positive_example(
    instance: SlotFeatures,
    templates: Dict[str, list],
    type2role: Dict[str, list],
    n: int = 1,
    label2id=None,
):
    """Generates an entailment example from a positive argument instance by sampling a random template
    from the correct argument role. Satisfies type constraints.
    """
    if instance.role in ["no_relation", "OOR"]:
        return []
    templates_ = templates[instance.role]
    return [
        MNLIInputFeatures(
            premise=instance.context,
            hypothesis=template.format(
                trg=instance.trigger,
                obj=instance.arg,
                trg_subtype=instance.trigger_type.split(".")[1],
            ),
            label=label2id["entailment"],
        )
        for template in random.sample(templates_, k=min(n, len(templates_)))
    ]


def generate_neutral_example(
    instance: SlotFeatures,
    templates: Dict[str, list],
    type2role: Dict[str, list],
    n: int = 1,
    label2id=None,
):
    """Generates a neutral example from a positive argument instance by sampling a random template
    from a random incorrect argument role. Satisfies type constraints.
    """
    if instance.role in ["no_relation", "OOR"]:
        return []

    posible_fake_roles = list(set(type2role[instance.pair_type]) - set([instance.role]))
    if not len(posible_fake_roles):
        return []
    templates_ = []
    for role in posible_fake_roles:
        templates_.extend(templates[role])
    return [
        MNLIInputFeatures(
            premise=instance.context,
            hypothesis=template.format(
                trg=instance.trigger,
                obj=instance.arg,
                trg_subtype=instance.trigger_type.split(".")[1],
            ),
            label=label2id["neutral"],
        )
        for template in random.sample(templates_, k=min(n, len(templates_)))
    ]


def generate_negative_example(
    instance: SlotFeatures,
    templates: Dict[str, list],
    type2role: Dict[str, list],
    n: int = 1,
    label2id=None,
):
    """Generates a contradiction example from a negative argument instance by sampling a random template
    from a positive argument role. Satisfies type constraints.
    """
    if instance.role not in ["no_relation"]:
        return []

    posible_fake_roles = type2role[instance.pair_type]
    if not len(posible_fake_roles):
        return []
    templates_ = []
    for role in posible_fake_roles:
        templates_.extend(templates[role])
    return [
        MNLIInputFeatures(
            premise=instance.context,
            hypothesis=template.format(
                trg=instance.trigger,
                obj=instance.arg,
                trg_subtype=instance.trigger_type.split(".")[1],
            ),
            label=label2id["contradiction"],
        )
        for template in random.sample(templates_, k=min(n, len(templates_)))
    ]


def to_nli(
    instance: SlotFeatures,
    templates: Dict[str, list],
    type2role: Dict[str, list],
    negn: int = 1,
    posn: int = 1,
    label2id=None,
):
    nli_examples = [
        *generate_positive_example(
            instance, templates, type2role, n=posn, label2id=label2id
        ),
        *generate_neutral_example(
            instance, templates, type2role, n=negn, label2id=label2id
        ),
        *generate_negative_example(
            instance, templates, type2role, n=negn, label2id=label2id
        ),
    ]

    return nli_examples


def main(opt):

    if opt.model_name_or_path:
        model_config = AutoConfig.from_pretrained(opt.model_name_or_path)
        label2id = {key.lower(): value for key, value in model_config.label2id.items()}

        if "not_entailment" in label2id:
            label2id = {
                "entailment": label2id["entailment"],
                "neutral": label2id["not_entailment"],
                "contradiction": label2id["not_entailment"],
            }
    else:
        label2id = {"entailment": 2, "neutral": 1, "contradiction": 0}

    with open(opt.config_file) as f:
        config = json.load(f)[0]

    dataset = WikiEventsArgumentDataset(
        opt.input_file,
        create_negatives=True,
        max_sentence_distance=config.get("max_sentence_distance", None),
        mark_trigger=True,
    )

    type2role = defaultdict(list)
    for role, type_pair_list in config["valid_conditions"].items():
        if role not in config["labels"]:
            continue
        for type_pair in type_pair_list:
            type2role[type_pair].append(role)

    nli_examples = []
    for instance in dataset:
        nli_examples.extend(
            to_nli(
                instance,
                config["template_mapping"],
                type2role,
                opt.negn,
                opt.posn,
                label2id=label2id,
            )
        )

    with open(opt.output_file, "wt") as f:
        for example in nli_examples:
            f.write(f"{json.dumps(example.__dict__)}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
