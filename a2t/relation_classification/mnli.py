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

@dataclass
class REInputFeatures:
    subj: str
    obj: str
    context: str
    label: str = None


class _NLIRelationClassifier(Classifier):

    def __init__(self, labels: List[str], *args, pretrained_model: str = 'roberta-large-mnli', use_cuda=True,
                 entailment_position=2, half=False, verbose=True, negative_threshold=.5, negative_idx=0, max_activations=3, **kwargs):
        super().__init__(labels, pretrained_model=pretrained_model, use_cuda=use_cuda, verbose=verbose, half=half)
        self.ent_pos = entailment_position
        self.cont_pos = -1 if self.ent_pos == 0 else 0
        self.negative_threshold = negative_threshold
        self.negative_idx = negative_idx
        self.max_activations = max_activations

    def _initialize(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)

    def _run_batch(self, batch, multiclass=False):
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(batch, padding=True)
            input_ids = torch.tensor(input_ids['input_ids']).to(self.device)
            output = self.model(input_ids)[0].detach().cpu().numpy()
            if multiclass:
                #print(torch.exp(output).shape, torch.exp(output[:, [self.cont_pos, self.ent_pos]]).sum(-1).view(-1, 1).shape)
                output = np.exp(output) / np.exp(output).sum(-1, keepdims=True) # np.exp(output[..., [self.cont_pos, self.ent_pos]]).sum(-1, keepdims=True)
            output = output[..., self.ent_pos].reshape(input_ids.shape[0] // len(self.labels), -1)
            #output = output.detach().cpu().numpy()

        return output

    def __call__(self, features: List[REInputFeatures], batch_size: int = 1, multiclass=False):
        if not isinstance(features, list):
            features = [features]

        batch, outputs = [], []
        for i, feature in tqdm(enumerate(features), total=len(features)):
            # sentences = [f"{feature.context} {self.tokenizer.sep_token} {feature.subj} {label_template} {feature.obj}." 
            #              for label_template in self.labels]
            sentences = [f"{feature.context} {self.tokenizer.sep_token} {label_template.format(subj=feature.subj, obj=feature.obj)}." 
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

    def _apply_negative_threshold(self, probs):
        activations = (probs >= self.negative_threshold).sum(-1).astype(np.int)
        idx = np.logical_or(activations == 0, activations >= self.max_activations) # If there are no activations then is a negative example, if there are too many, then is a noisy example 
        probs[idx, self.negative_idx] = 1.00
        return probs


class NLIRelationClassifier(_NLIRelationClassifier):

    def __init__(self, labels: List[str], *args, pretrained_model: str = 'roberta-large-mnli', **kwargs):
        super(NLITopicClassifier, self).__init__(labels, *args, pretrained_model=pretrained_model, **kwargs)

    def __call__(self, features: List[REInputFeatures], batch_size: int = 1, multiclass=True):
        outputs = super().__call__(features=features, batch_size=batch_size, multiclass=multiclass)
        outputs = np_softmax(outputs) if not multiclass else outputs
        outputs = self._apply_negative_threshold(outputs)

        return outputs


class NLIRelationClassifierWithMappingHead(_NLIRelationClassifier):

    def __init__(self, labels: List[str], template_mapping: Dict[str, str],
                 pretrained_model: str = 'roberta-large-mnli',  *args, **kwargs):
        self.new_topics = list(template_mapping.keys())
        self.target_labels = labels
        self.new_labels2id = {t: i for i, t in enumerate(self.new_topics)}
        self.mapping = defaultdict(list)
        for key, value in template_mapping.items():
            self.mapping[value].append(self.new_labels2id[key])

        super().__init__(self.new_topics, *args, pretrained_model=pretrained_model, **kwargs)

    def __call__(self, features: List[REInputFeatures], batch_size=1, multiclass=True):
        outputs = super().__call__(features, batch_size, multiclass)
        outputs = np.hstack([np.max(outputs[:, self.mapping[label]], axis=-1, keepdims=True) 
                             if label in self.mapping else np.zeros((outputs.shape[0], 1))
                             for label in self.target_labels])
        outputs = np_softmax(outputs) if not multiclass else outputs
        outputs = self._apply_negative_threshold(outputs)

        return outputs


if __name__ == "__main__":

    labels = [
        'no_relation', 
        'org:alternate_names', 
        'org:city_of_headquarters', 
        'org:country_of_headquarters', 
        'org:dissolved', 
        'org:founded', 
        'org:founded_by', 
        'org:member_of', 
        'org:members', 
        'org:number_of_employees/members', 
        'org:parents', 
        'org:political/religious_affiliation', 
        'org:shareholders', 
        'org:stateorprovince_of_headquarters', 
        'org:subsidiaries', 
        'org:top_members/employees', 
        'org:website', 
        'per:age', 
        'per:alternate_names', 
        'per:cause_of_death', 
        'per:charges', 
        'per:children', 
        'per:cities_of_residence', 
        'per:city_of_birth', 
        'per:city_of_death', 
        'per:countries_of_residence', 
        'per:country_of_birth', 
        'per:country_of_death', 
        'per:date_of_birth', 
        'per:date_of_death', 
        'per:employee_of', 
        'per:origin', 
        'per:other_family', 
        'per:parents', 
        'per:religion', 
        'per:schools_attended', 
        'per:siblings', 
        'per:spouse', 
        'per:stateorprovince_of_birth', 
        'per:stateorprovince_of_death', 
        'per:stateorprovinces_of_residence', 
        'per:title'
    ]#['no_relation', 'per:city_of_death', 'org:founded_by']
    template_mapping = {
        "{subj} is also known as {obj}": "org:alternate_names",
        "{subj} has a headquarter in {obj} city": "org:city_of_headquarters",
        "{subj} has a headquarter in {obj} country": "org:country_of_headquarters",
        "{subj} was dissolved by {obj}": "org:dissolved",
        "{subj} has founded {obj}": "org:founded",
        "{subj} is founded by {obj}": "org:founded_by",
        "{subj} is member of {obj}": "org:member_of",
        "{subj} is form by {obj}": "org:members",
        "{subj} has {obj} members": "org:number_of_employees/members",
        "{subj} has die in {obj}": "per:city_of_death",
        "{subj} is not related to {obj}": "no_relation",
        "{subj} has no relation with {obj}": "no_relation",
        "{subj} and {obj} are not related.": "no_relation"
        # 'has die in': 'per:city_of_death',
        # 'is founded by': 'org:founded_by'
    }

    clf = NLIRelationClassifierWithMappingHead(
        labels=labels, template_mapping=template_mapping,
        pretrained_model='facebook/bart-large-mnli'
    )

    features = [
        REInputFeatures(subj='Billy Mays', obj='Tampa', context='Billy Mays, the bearded, boisterous pitchman who, as the undisputed king of TV yell and sell, became an unlikely pop culture icon, died at his home in Tampa, Fla, on Sunday', label='per:city_of_death'),
        REInputFeatures(subj='Old Lane Partners', obj='Pandit', context='Pandit worked at the brokerage Morgan Stanley for about 11 years until 2005, when he and some Morgan Stanley colleagues quit and later founded the hedge fund Old Lane Partners.', label='org:founded_by'),
        #REInputFeatures(subj='He', obj='University of Maryland in College Park', context='He received an undergraduate degree from Morgan State University in 1950 and applied for admission to graduate school at the University of Maryland in College Park.', label='no_relation')
    ]

    predictions = clf(features, multiclass=True)
    for pred in predictions:
        pprint(sorted(list(zip(pred, labels)), reverse=True))