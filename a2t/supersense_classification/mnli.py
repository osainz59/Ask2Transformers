from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from a2t.base import Classifier, np_softmax
from a2t.supersense_classification.wordnet_lexnames import (
    WORDNET_LEXNAMES_BY_POS, WORDNET_LEXNAMES_TO_DEFINITIONS, WORDNET_LEXNAMES
)


class _NLISuperSenseClassifier(Classifier):

    def __init__(self, *args, senses: List[str] = WORDNET_LEXNAMES,  pretrained_model:str = 'roberta-large-mnli',
                 use_cuda=True, query_phrase="The semantic field is", entailment_position=2, half=False,
                 verbose=True, **kwargs):
        super().__init__(senses, pretrained_model, use_cuda=use_cuda, verbose=verbose, half=half)
        self.query_phrase = query_phrase
        self.ent_pos = entailment_position

    def _initialize(self, pretrained_model='roberta-large-mnli'):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)

    def _run_batch(self, batch):
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(batch, padding=True)
            input_ids = torch.tensor(input_ids['input_ids']).to(self.device)
            output = self.model(input_ids)[0][:, self.ent_pos].view(input_ids.shape[0] // len(self.labels), -1)
            output = output.detach().cpu().numpy()

        return output

    def __call__(self, contexts: List[str], batch_size: int = 1):
        if not isinstance(contexts, list):
            contexts = [contexts]

        batch, outputs = [], []
        for i, context in tqdm(enumerate(contexts), total=len(contexts)):
            sentences = [f"{context} {self.tokenizer.sep_token} {self.query_phrase} {WORDNET_LEXNAMES_TO_DEFINITIONS[lexname]}." for lexname in
                         self.labels]
            batch.extend(sentences)

            if (i + 1) % batch_size == 0:
                output = self._run_batch(batch)
                outputs.append(output)
                batch = []

        if len(batch) > 0:
            output = self._run_batch(batch)
            outputs.append(output)

        outputs = np.vstack(outputs)

        return outputs


class NLISuperSenseClassifier(_NLISuperSenseClassifier):

    def __init__(self, *args, pretrained_model: str = 'roberta-large-mnli', **kwargs):
        super(NLISuperSenseClassifier, self).__init__(senses=WORDNET_LEXNAMES, pretrained_model=pretrained_model,
                                                      *args, **kwargs)

    def __call__(self, contexts: List[str], batch_size: int = 1):
        outputs = super().__call__(contexts=contexts, batch_size=batch_size)
        outputs = np_softmax(outputs)

        return outputs


class POSAwareNLISuperSenseClassifier(_NLISuperSenseClassifier):
    """ Part-Of-Speech Aware NLI SuperSense Classifier
    TODO: Test

    This classifier allows to discard those classes that are not related to the actual POS of the target word.

    """
    pass

    def __init__(self, **kwargs):
        self.sense2id = {sense: i for i, sense in enumerate(WORDNET_LEXNAMES)}
        self.sense_pos_mask = {}
        for sense, mask in WORDNET_LEXNAMES_BY_POS.items():
            self.sense_pos_mask[sense] = np.zeros(len(WORDNET_LEXNAMES))
            mask = [self.sense2id[s] for s in mask]
            self.sense_pos_mask[sense][mask] = 1.

        def to_np_mask(pos):
            return self.sense_pos_mask[pos]

        self.to_np_mask = np.vectorize(to_np_mask)

        super(POSAwareNLISuperSenseClassifier, self).__init__(labels=WORDNET_LEXNAMES, **kwargs)

    def __call__(self, contexts: List[str], pos_tags: List[str], batch_size: int = 1):
        assert len(contexts) == len(pos_tags)

        outputs = super().__call__(contexts=contexts, batch_size=batch_size)
        outputs = np.multiply(outputs, self.to_np_mask(pos_tags))
        outputs = np_softmax(outputs)

        return outputs