from collections import defaultdict
from pprint import pprint
from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    T5ForConditionalGeneration,
)

from a2t.base import Classifier, np_softmax


@dataclass
class REInputFeatures:
    subj: str
    obj: str
    context: str
    pair_type: str = None
    label: str = None


class _NLIRelationClassifier(Classifier):
    def __init__(
        self,
        labels: List[str],
        *args,
        pretrained_model: str = "roberta-large-mnli",
        use_cuda=True,
        half=False,
        verbose=True,
        negative_threshold=0.95,
        negative_idx=0,
        max_activations=np.inf,
        valid_conditions=None,
        **kwargs,
    ):
        super().__init__(
            labels,
            pretrained_model=pretrained_model,
            use_cuda=use_cuda,
            verbose=verbose,
            half=half,
        )
        # self.ent_pos = entailment_position
        # self.cont_pos = -1 if self.ent_pos == 0 else 0
        self.negative_threshold = negative_threshold
        self.negative_idx = negative_idx
        self.max_activations = max_activations
        self.n_rel = len(labels)
        # for label in labels:
        #     assert '{subj}' in label and '{obj}' in label

        if valid_conditions:
            self.valid_conditions = {}
            rel2id = {r: i for i, r in enumerate(labels)}
            self.n_rel = len(rel2id)
            for relation, conditions in valid_conditions.items():
                if relation not in rel2id:
                    continue
                for condition in conditions:
                    if condition not in self.valid_conditions:
                        self.valid_conditions[condition] = np.zeros(self.n_rel)
                        self.valid_conditions[condition][rel2id["no_relation"]] = 1.0
                    self.valid_conditions[condition][rel2id[relation]] = 1.0

        else:
            self.valid_conditions = None

        def idx2label(idx):
            return self.labels[idx]

        self.idx2label = np.vectorize(idx2label)

    def _initialize(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.ent_pos = self.config.label2id.get("ENTAILMENT", self.config.label2id.get("entailment", None))
        if self.ent_pos is None:
            raise ValueError("The model config must contain ENTAILMENT label in the label2id dict.")
        else:
            self.ent_pos = int(self.ent_pos)

    def _run_batch(self, batch, multiclass=False):
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(batch, padding=True, truncation=True)
            input_ids = torch.tensor(input_ids["input_ids"]).to(self.device)
            output = self.model(input_ids)[0].detach().cpu().numpy()
            if multiclass:
                output = np.exp(output) / np.exp(output).sum(
                    -1, keepdims=True
                )  # np.exp(output[..., [self.cont_pos, self.ent_pos]]).sum(-1, keepdims=True)
            output = output[..., self.ent_pos].reshape(input_ids.shape[0] // len(self.labels), -1)

        return output

    def __call__(
        self,
        features: List[REInputFeatures],
        batch_size: int = 1,
        multiclass=False,
    ):
        if not isinstance(features, list):
            features = [features]

        batch, outputs = [], []
        for i, feature in tqdm(enumerate(features), total=len(features)):
            sentences = [
                f"{feature.context} {self.tokenizer.sep_token} {label_template.format(subj=feature.subj, obj=feature.obj)}."
                for label_template in self.labels
            ]
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
        idx = np.logical_or(
            activations == 0, activations >= self.max_activations
        )  # If there are no activations then is a negative example, if there are too many, then is a noisy example
        probs[idx, self.negative_idx] = 1.00
        return probs

    def _apply_valid_conditions(self, probs, features: List[REInputFeatures]):
        mask_matrix = np.stack(
            [self.valid_conditions.get(feature.pair_type, np.zeros(self.n_rel)) for feature in features],
            axis=0,
        )
        probs = probs * mask_matrix

        return probs

    def predict(
        self,
        contexts: List[str],
        batch_size: int = 1,
        return_labels: bool = True,
        return_confidences: bool = False,
        topk: int = 1,
    ):
        output = self(contexts, batch_size)
        topics = np.argsort(output, -1)[:, ::-1][:, :topk]
        if return_labels:
            topics = self.idx2label(topics)
        if return_confidences:
            topics = np.stack((topics, np.sort(output, -1)[:, ::-1][:, :topk]), -1).tolist()
            topics = [
                [(int(label), float(conf)) if not return_labels else (label, float(conf)) for label, conf in row]
                for row in topics
            ]
        else:
            topics = topics.tolist()
        if topk == 1:
            topics = [row[0] for row in topics]

        return topics


class _GenerativeNLIRelationClassifier(_NLIRelationClassifier):
    """_GenerativeNLIRelationClassifier

    This class is intended to be use with T5 like NLI models.

    TODO: Test
    """

    def _initialize(self, pretrained_model):

        if "t5" not in pretrained_model:
            raise NotImplementedError(
                f"This implementation is not available for {pretrained_model} yet. "
                "Use a t5-[small-base-large] model instead."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self._get_entailment_neutral_contradiction_token_id()

    def _run_batch(self, batch, multiclass=False):
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(batch, padding=True, truncation=True)
            input_ids = torch.tensor(input_ids["input_ids"]).to(self.device)
            decoder_input_ids = self.tokenizer.batch_encode_plus(len(batch) * ["<pad>"], padding=True, truncation=True)
            decoder_input_ids = torch.tensor(decoder_input_ids["input_ids"]).to(self.device)
            output = self.model(input_ids, decoder_input_ids=decoder_input_ids)[0].detach().cpu().numpy()
            output = self._vocab_to_class_logits(output)
            if multiclass:
                output = np.exp(output) / np.exp(output).sum(
                    -1, keepdims=True
                )  # np.exp(output[..., [self.cont_pos, self.ent_pos]]).sum(-1, keepdims=True)
            output = output[..., 0].reshape(input_ids.shape[0] // len(self.labels), -1)

        return output

    def _get_entailment_neutral_contradiction_token_id(self):
        class_ids = self.tokenizer(["entailment", "neutral", "contradiction"]).input_ids
        (
            self.entailment_token_id,
            self.neutral_token_id,
            self.contradiction_token_id,
        ) = [ids[0] for ids in class_ids]
        assert (
            (self.entailment_token_id != self.neutral_token_id)
            and (self.entailment_token_id != self.contradiction_token_id)
            and (self.neutral_token_id != self.contradiction_token_id)
        )

    def _vocab_to_class_logits(self, outputs):
        class_logits = outputs[
            :,
            0,
            [
                self.entailment_token_id,
                self.neutral_token_id,
                self.contradiction_token_id,
            ],
        ]
        return class_logits


class NLIRelationClassifier(_NLIRelationClassifier):
    def __init__(
        self,
        labels: List[str],
        *args,
        pretrained_model: str = "roberta-large-mnli",
        **kwargs,
    ):
        super().__init__(labels, *args, pretrained_model=pretrained_model, **kwargs)

    def __call__(
        self,
        features: List[REInputFeatures],
        batch_size: int = 1,
        multiclass=True,
    ):
        outputs = super().__call__(features=features, batch_size=batch_size, multiclass=multiclass)
        outputs = np_softmax(outputs) if not multiclass else outputs
        outputs = self._apply_negative_threshold(outputs)

        return outputs


class NLIRelationClassifierWithMappingHead(_NLIRelationClassifier):
    def __init__(
        self,
        labels: List[str],
        template_mapping: Dict[str, str],
        pretrained_model: str = "roberta-large-mnli",
        valid_conditions: Dict[str, list] = None,
        *args,
        **kwargs,
    ):

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

        super().__init__(
            self.new_topics,
            *args,
            pretrained_model=pretrained_model,
            valid_conditions=None,
            **kwargs,
        )

        if valid_conditions:
            self.valid_conditions = {}
            rel2id = {r: i for i, r in enumerate(labels)}
            self.n_rel = len(rel2id)
            for relation, conditions in valid_conditions.items():
                if relation not in rel2id:
                    continue
                for condition in conditions:
                    if condition not in self.valid_conditions:
                        self.valid_conditions[condition] = np.zeros(self.n_rel)
                        self.valid_conditions[condition][rel2id["no_relation"]] = 1.0
                    self.valid_conditions[condition][rel2id[relation]] = 1.0

        else:
            self.valid_conditions = None

        def idx2label(idx):
            return self.target_labels[idx]

        self.idx2label = np.vectorize(idx2label)

    def __call__(self, features: List[REInputFeatures], batch_size=1, multiclass=True):
        outputs = super().__call__(features, batch_size, multiclass)
        outputs = np.hstack(
            [
                np.max(outputs[:, self.mapping[label]], axis=-1, keepdims=True)
                if label in self.mapping
                else np.zeros((outputs.shape[0], 1))
                for label in self.target_labels
            ]
        )
        outputs = np_softmax(outputs) if not multiclass else outputs

        if self.valid_conditions:
            outputs = self._apply_valid_conditions(outputs, features)

        outputs = self._apply_negative_threshold(outputs)

        return outputs


class GenerativeNLIRelationClassifier(_GenerativeNLIRelationClassifier, NLIRelationClassifier):
    pass


class GenerativeNLIRelationClassifierWithMappingHead(_GenerativeNLIRelationClassifier, NLIRelationClassifierWithMappingHead):
    pass


if __name__ == "__main__":

    labels = [
        "no_relation",
        "org:alternate_names",
        "org:city_of_headquarters",
        "org:country_of_headquarters",
        "org:dissolved",
        "org:founded",
        "org:founded_by",
        "org:member_of",
        "org:members",
        "org:number_of_employees/members",
        "org:parents",
        "org:political/religious_affiliation",
        "org:shareholders",
        "org:stateorprovince_of_headquarters",
        "org:subsidiaries",
        "org:top_members/employees",
        "org:website",
        "per:age",
        "per:alternate_names",
        "per:cause_of_death",
        "per:charges",
        "per:children",
        "per:cities_of_residence",
        "per:city_of_birth",
        "per:city_of_death",
        "per:countries_of_residence",
        "per:country_of_birth",
        "per:country_of_death",
        "per:date_of_birth",
        "per:date_of_death",
        "per:employee_of",
        "per:origin",
        "per:other_family",
        "per:parents",
        "per:religion",
        "per:schools_attended",
        "per:siblings",
        "per:spouse",
        "per:stateorprovince_of_birth",
        "per:stateorprovince_of_death",
        "per:stateorprovinces_of_residence",
        "per:title",
    ]
    template_mapping = {
        "per:alternate_names": ["{subj} is also known as {obj}"],
        "per:date_of_birth": [
            "{subj}’s birthday is on {obj}",
            "{subj} was born in {obj}",
        ],
        "per:age": ["{subj} is {obj} years old"],
        "per:country_of_birth": ["{subj} was born in {obj}"],
        "per:stateorprovince_of_birth": ["{subj} was born in {obj}"],
        "per:city_of_birth": ["{subj} was born in {obj}"],
        "per:origin": ["{obj} is the nationality of {subj}"],
        "per:date_of_death": ["{subj} died in {obj}"],
        "per:country_of_death": ["{subj} died in {obj}"],
        "per:stateorprovince_of_death": ["{subj} died in {obj}"],
        "per:city_of_death": ["{subj} died in {obj}"],
        "per:cause_of_death": ["{obj} is the cause of {subj}’s death"],
        "per:countries_of_residence": [
            "{subj} lives in {obj}",
            "{subj} has a legal order to stay in {obj}",
        ],
        "per:stateorprovinces_of_residence": [
            "{subj} lives in {obj}",
            "{subj} has a legal order to stay in {obj}",
        ],
        "per:cities_of_residence": [
            "{subj} lives in {obj}",
            "{subj} has a legal order to stay in {obj}",
        ],
        "per:schools_attended": [
            "{subj} studied in {obj}",
            "{subj} graduated from {obj}",
        ],
        "per:title": ["{subj} is a {obj}"],
        "per:employee_of": [
            "{subj} is member of {obj}",
            "{subj} is an employee of {obj}",
        ],
        "per:religion": [
            "{subj} belongs to {obj} religion",
            "{obj} is the religion of {subj}",
            "{subj} believe in {obj}",
        ],
        "per:spouse": [
            "{subj} is the spouse of {obj}",
            "{subj} is the wife of {obj}",
            "{subj} is the husband of {obj}",
        ],
        "per:children": [
            "{subj} is the parent of {obj}",
            "{subj} is the mother of {obj}",
            "{subj} is the father of {obj}",
            "{subj} is the son of {obj}",
            "{subj} is the daughter of {obj}",
        ],
        "per:siblings": [
            "{subj} and {obj} are siblings",
            "{subj} is brother of {obj}",
            "{subj} is sister of {obj}",
        ],
        "per:other_family": [
            "{subj} and {obj} are family",
            "{subj} is a brother in law of {obj}",
            "{subj} is a sister in law of {obj}",
            "{subj} is the cousin of {obj}",
            "{subj} is the uncle of {obj}",
            "{subj} is the aunt of {obj}",
            "{subj} is the grandparent of {obj}",
            "{subj} is the grandmother of {obj}",
            "{subj} is the grandson of {obj}",
            "{subj} is the granddaughter of {obj}",
        ],
        "per:charges": [
            "{subj} was convicted of {obj}",
            "{obj} are the charges of {subj}",
        ],
        "org:alternate_names": ["{subj} is also known as {obj}"],
        "org:political/religious_affiliation": [
            "{subj} has political affiliation with {obj}",
            "{subj} has religious affiliation with {obj}",
        ],
        "org:top_members/employees": [
            "{obj} is a high level member of {subj}",
            "{obj} is chairman of {subj}",
            "{obj} is president of {subj}",
            "{obj} is director of {subj}",
        ],
        "org:number_of_employees/members": [
            "{subj} employs nearly {obj} people",
            "{subj} has about {obj} employees",
        ],
        "org:members": ["{obj} is member of {subj}", "{obj} joined {subj}"],
        "org:member_of": ["{subj} is member of {obj}", "{subj} joined {obj}"],
        "org:subsidiaries": [
            "{obj} is a subsidiary of {subj}",
            "{obj} is a branch of {subj}",
        ],
        "org:parents": [
            "{subj} is a subsidiary of {obj}",
            "{subj} is a branch of {obj}",
        ],
        "org:founded_by": [
            "{subj} was founded by {obj}",
            "{obj} founded {subj}",
        ],
        "org:founded": [
            "{subj} was founded in {obj}",
            "{subj} was formed in {obj}",
        ],
        "org:dissolved": [
            "{subj} existed until {obj}",
            "{subj} disbanded in {obj}",
            "{subj} dissolved in {obj}",
        ],
        "org:country_of_headquarters": [
            "{subj} has its headquarters in {obj}",
            "{subj} is located in {obj}",
        ],
        "org:stateorprovince_of_headquarters": [
            "{subj} has its headquarters in {obj}",
            "{subj} is located in {obj}",
        ],
        "org:city_of_headquarters": [
            "{subj} has its headquarters in {obj}",
            "{subj} is located in {obj}",
        ],
        "org:shareholders": ["{obj} holds shares in {subj}"],
        "org:website": [
            "{obj} is the URL of {subj}",
            "{obj} is the website of {subj}",
        ],
    }

    rules = {
        "per:alternate_names": ["PERSON:PERSON", "PERSON:MISC"],
        "per:date_of_birth": ["PERSON:DATE"],
        "per:age": ["PERSON:NUMBER", "PERSON:DURATION"],
        "per:country_of_birth": ["PERSON:COUNTRY"],
        "per:stateorprovince_of_birth": ["PERSON:STATE_OR_PROVINCE"],
        "per:city_of_birth": ["PERSON:CITY"],
        "per:origin": [
            "PERSON:NATIONALITY",
            "PERSON:COUNTRY",
            "PERSON:LOCATION",
        ],
        "per:date_of_death": ["PERSON:DATE"],
        "per:country_of_death": ["PERSON:COUNTRY"],
        "per:stateorprovince_of_death": ["PERSON:STATE_OR_PROVICE"],
        "per:city_of_death": ["PERSON:CITY", "PERSON:LOCATION"],
        "per:cause_of_death": ["PERSON:CAUSE_OF_DEATH"],
        "per:countries_of_residence": ["PERSON:COUNTRY", "PERSON:NATIONALITY"],
        "per:stateorprovinces_of_residence": ["PERSON:STATE_OR_PROVINCE"],
        "per:cities_of_residence": ["PERSON:CITY", "PERSON:LOCATION"],
        "per:schools_attended": ["PERSON:ORGANIZATION"],
        "per:title": ["PERSON:TITLE"],
        "per:employee_of": ["PERSON:ORGANIZATION"],
        "per:religion": ["PERSON:RELIGION"],
        "per:spouse": ["PERSON:PERSON"],
        "per:children": ["PERSON:PERSON"],
        "per:siblings": ["PERSON:PERSON"],
        "per:other_family": ["PERSON:PERSON"],
        "per:charges": ["PERSON:CRIMINAL_CHARGE"],
        "org:alternate_names": [
            "ORGANIZATION:ORGANIZATION",
            "ORGANIZATION:MISC",
        ],
        "org:political/religious_affiliation": [
            "ORGANIZATION:RELIGION",
            "ORGANIZATION:IDEOLOGY",
        ],
        "org:top_members/employees": ["ORGANIZATION:PERSON"],
        "org:number_of_employees/members": ["ORGANIZATION:NUMBER"],
        "org:members": ["ORGANIZATION:ORGANIZATION", "ORGANIZATION:COUNTRY"],
        "org:member_of": [
            "ORGANIZATION:ORGANIZATION",
            "ORGANIZATION:COUNTRY",
            "ORGANIZATION:LOCATION",
            "ORGANIZATION:STATE_OR_PROVINCE",
        ],
        "org:subsidiaries": [
            "ORGANIZATION:ORGANIZATION",
            "ORGANIZATION:LOCATION",
        ],
        "org:parents": ["ORGANIZATION:ORGANIZATION", "ORGANIZATION:COUNTRY"],
        "org:founded_by": ["ORGANIZATION:PERSON"],
        "org:founded": ["ORGANIZATION:DATE"],
        "org:dissolved": ["ORGANIZATION:DATE"],
        "org:country_of_headquarters": ["ORGANIZATION:COUNTRY"],
        "org:stateorprovince_of_headquarters": ["ORGANIZATION:STATE_OR_PROVINCE"],
        "org:city_of_headquarters": [
            "ORGANIZATION:CITY",
            "ORGANIZATION:LOCATION",
        ],
        "org:shareholders": [
            "ORGANIZATION:PERSON",
            "ORGANIZATION:ORGANIZATION",
        ],
        "org:website": ["ORGANIZATION:URL"],
    }

    clf = NLIRelationClassifierWithMappingHead(
        labels=labels,
        template_mapping=template_mapping,
        pretrained_model="microsoft/deberta-v2-xlarge-mnli",
        valid_conditions=rules,
    )

    features = [
        REInputFeatures(
            subj="Billy Mays",
            obj="Tampa",
            pair_type="PERSON:CITY",
            context="Billy Mays, the bearded, boisterous pitchman who, as the undisputed king of TV yell and sell, became an unlikely pop culture icon, died at his home in Tampa, Fla, on Sunday",
            label="per:city_of_death",
        ),
        REInputFeatures(
            subj="Old Lane Partners",
            obj="Pandit",
            pair_type="ORGANIZATION:PERSON",
            context="Pandit worked at the brokerage Morgan Stanley for about 11 years until 2005, when he and some Morgan Stanley colleagues quit and later founded the hedge fund Old Lane Partners.",
            label="org:founded_by",
        ),
        # REInputFeatures(subj='He', obj='University of Maryland in College Park', context='He received an undergraduate degree from Morgan State University in 1950 and applied for admission to graduate school at the University of Maryland in College Park.', label='no_relation')
    ]

    predictions = clf(features, multiclass=True)
    for pred in predictions:
        pprint(sorted(list(zip(pred, labels)), reverse=True)[:5])
