from typing import List

import numpy as np

from a2t.relation_classification.mnli import NLIRelationClassifierWithMappingHead

TACRED_LABELS = [
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
]

TACRED_BASIC_LABELS = [
    'no_relation', 
    'org:dissolved', 
    'org:founded', 
    'org:founded_by', 
    'org:member_of', 
    'org:members', 
    'org:number_of_employees/members', 
    'org:parents', 
    'org:political/religious_affiliation', 
    'org:shareholders', 
    'org:subsidiaries', 
    'org:top_members/employees', 
    'org:website', 
    'per:age', 
    'per:cause_of_death', 
    'per:charges', 
    'per:children', 
    'per:date_of_birth', 
    'per:date_of_death', 
    'per:employee_of', 
    'per:other_family', 
    'per:parents', 
    'per:religion', 
    'per:schools_attended', 
    'per:siblings', 
    'per:spouse', 
    'per:title',
    'per:place_of_birth',
    'per:place_of_death',
    'per:place_of_residence',
    'org:place_of_headquarters',
    'both:alternate_names'
]

TACRED_BASIC_LABELS_MAPPING = {
    # Place of birth
    'per:city_of_birth': 'per:place_of_birth',
    'per:country_of_birth': 'per:place_of_birth',
    'per:stateorprovince_of_birth': 'per:place_of_birth',
    # Origin can be see as a country/place of birth
    'per:origin': 'per:place_of_birth',
    # Place of death
    'per:city_of_death': 'per:place_of_death',
    'per:country_of_death': 'per:place_of_death',
    'per:stateorprovince_of_death': 'per:place_of_death',
    # Place of residence
    'per:cities_of_residence': 'per:place_of_residence',
    'per:countries_of_residence': 'per:place_of_residence',
    'per:stateorprovinces_of_residence': 'per:place_of_residence',
    # Place of headquarters
    'org:city_of_headquarters': 'org:place_of_headquarters',
    'org:country_of_headquarters': 'org:place_of_headquarters',
    'org:stateorprovince_of_headquarters': 'org:place_of_headquarters',
    # Alternative names
    'per:alternate_names': 'both:alternate_names',
    'org:alternate_names': 'both:alternate_names',
}

TACRED_LABEL_TEMPLATES = {
    '{subj} has die in {obj}': 'per:city_of_death',
    '{subj} is founded by {obj}': 'org:founded_by'
}

TACRED_VALID_CONDITIONS = {
    "per:alternate_names": [
        "PERSON:PERSON"
    ]
}


class TACREDClassifier(NLIRelationClassifierWithMappingHead):

    def __init__(self, **kwargs):
        super(TACREDClassifier, self).__init__(
            pretrained_model='roberta-large-mnli', labels=TACRED_LABELS, template_mapping=TACRED_LABEL_TEMPLATES,
            entailment_position=2, **kwargs)

        def idx2topic(idx):
            return TACRED_LABELS[idx]

        self.idx2topic = np.vectorize(idx2topic)
        
    def predict_topics(self, contexts: List[str], batch_size: int = 1, return_labels: bool = True,
                       return_confidences: bool = False, topk: int = 1):
        output = self(contexts, batch_size)
        topics = np.argsort(output, -1)[:, ::-1][:, :topk]
        if return_labels:
            topics = self.idx2topic(topics)
        if return_confidences:
            topics = np.stack((topics, np.sort(output, -1)[:, ::-1][:, :topk]), -1).tolist()
            topics = [[(int(label), conf) if not return_labels else (label, conf) for label, conf in row]
                      for row in topics]
        else:
            topics = topics.tolist()
        if topk == 1:
            topics = [row[0] for row in topics]

        return topics
