import sys
from collections import defaultdict
from pprint import pprint
from typing import Dict, List
from dataclasses import dataclass
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from a2t.base import Classifier, np_softmax, np_sigmoid
from a2t.relation_classification.mnli import _NLIRelationClassifier
from a2t.slot_classification.data import SlotFeatures


class _NLISlotClassifier(_NLIRelationClassifier):

    def __call__(self, features: List[SlotFeatures], batch_size: int = 1, multiclass=False):
        if not isinstance(features, list):
            features = [features]

        batch, outputs = [], []
        for i, feature in tqdm(enumerate(features), total=len(features)):
            sentences = [f"{feature.context} {self.tokenizer.sep_token} {label_template.format(trg=feature.trigger, obj=feature.arg, trg_subtype=feature.trigger_type.split('.')[1])}." 
                         for label_template in self.labels]
            batch.extend(sentences)

            if (i + 1) % batch_size == 0:
                output = self._run_batch(batch, multiclass=multiclass)
                outputs.append(output)
                batch = []

        if len(batch) > 0:
            output = self._run_batch(batch, multiclass=multiclass)
            outputs.append(output)

        outputs = np.vstack(outputs)

        return outputs

class NLISlotClassifier(_NLISlotClassifier):

    def __init__(self, labels: List[str], *args, pretrained_model: str = 'roberta-large-mnli', **kwargs):
        super(NLITopicClassifier, self).__init__(labels, *args, pretrained_model=pretrained_model, **kwargs)

    def __call__(self, features: List[SlotFeatures], batch_size: int = 1, multiclass=True):
        outputs = super().__call__(features=features, batch_size=batch_size, multiclass=multiclass)
        outputs = np_softmax(outputs) if not multiclass else outputs
        outputs = self._apply_negative_threshold(outputs)

        return outputs

class NLISlotClassifierWithMappingHead(_NLISlotClassifier):

    def __init__(self, labels: List[str], template_mapping: Dict[str, str],
                 pretrained_model: str = 'roberta-large-mnli',
                 valid_conditions: Dict[str, list] = None, *args, **kwargs):

        self.template_mapping_reverse = defaultdict(list)
        for key, value in template_mapping.items():
            for v in value:
                self.template_mapping_reverse[v].append(key)
        self.new_topics = list(self.template_mapping_reverse.keys())

        self.target_labels = labels
        self.new_labels2id = {t: i for i, t in enumerate(self.new_topics)}
        self.mapping = defaultdict(list)
        for key, value in template_mapping.items():
            self.mapping[key].extend([self.new_labels2id[v] for v in value])

        super().__init__(self.new_topics, *args, pretrained_model=pretrained_model, 
                         valid_conditions=None, **kwargs)

        if valid_conditions:
            self.valid_conditions = {}
            rel2id = {r:i for i, r in enumerate(labels)}
            self.n_rel = len(rel2id)
            for relation, conditions in valid_conditions.items():
                if relation not in rel2id:
                    continue
                for condition in conditions:
                    if condition not in self.valid_conditions:
                        self.valid_conditions[condition] = np.zeros(self.n_rel)
                        self.valid_conditions[condition][rel2id['no_relation']] = 1.
                    self.valid_conditions[condition][rel2id[relation]] = 1.

        else:
            self.valid_conditions = None

    def __call__(self, features: List[SlotFeatures], batch_size=1, multiclass=True):
        outputs = super().__call__(features, batch_size, multiclass)
        outputs = np.hstack([np.max(outputs[:, self.mapping[label]], axis=-1, keepdims=True) 
                             if label in self.mapping else np.zeros((outputs.shape[0], 1))
                             for label in self.target_labels])
        outputs = np_softmax(outputs) if not multiclass else outputs

        if self.valid_conditions:
            outputs = self._apply_valid_conditions(outputs, features)

        outputs = self._apply_negative_threshold(outputs)

        return outputs


if __name__ == "__main__":
    model = NLISlotClassifierWithMappingHead(["test"], {"test":["{subj} and {obj}"]})