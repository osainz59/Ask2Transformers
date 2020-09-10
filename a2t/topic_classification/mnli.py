import sys
from collections import defaultdict
from pprint import pprint
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from a2t.topic_classification.base import TopicClassifier, np_softmax


class _NLITopicClassifier(TopicClassifier):

    def __init__(self, pretrained_model, topics, *args, use_cuda=True,
                 query_phrase="The domain of the sentence is about", entailment_position=2, 
                 half=False, verbose=True, **kwargs):
        super().__init__(pretrained_model, topics, use_cuda=use_cuda, verbose=verbose, half=half)
        self.query_phrase = query_phrase
        self.ent_pos = entailment_position

    def _initialize(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)

    def _run_batch(self, batch):
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(batch, padding=True)
            input_ids = torch.tensor(input_ids['input_ids']).to(self.device)
            output = self.model(input_ids)[0][:, self.ent_pos].view(input_ids.shape[0] // len(self.topics), -1)
            output = output.detach().cpu().numpy()

        return output

    def __call__(self, contexts: List[str], batch_size: int = 1):
        if not isinstance(contexts, list):
            contexts = [contexts]

        batch, outputs = [], []
        for i, context in tqdm(enumerate(contexts), total=len(contexts)):
            sentences = [f"{context} {self.tokenizer.sep_token} {self.query_phrase} \"{topic}\"." for topic in
                        self.topics]
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


class NLITopicClassifier(_NLITopicClassifier):

    def __init__(self, pretrained_model: str, topics: List[str], *args, **kwargs):
        super(NLITopicClassifier, self).__init__(pretrained_model, topics, *args, **kwargs)

    def __call__(self, contexts: List[str], batch_size: int = 1):
        outputs = super().__call__(contexts=contexts, batch_size=batch_size)
        outputs = np_softmax(outputs)

        return outputs


class NLITopicClassifierWithMappingHead(_NLITopicClassifier):

    def __init__(self, pretrained_model: str, topics: List[str], topic_mapping: Dict[str, str], *args, **kwargs):
        self.new_topics = list(topic_mapping.keys())
        self.target_topics = topics
        self.new_topics2id = {t: i for i, t in enumerate(self.new_topics)}
        self.mapping = defaultdict(list)
        for key, value in topic_mapping.items():
            self.mapping[value].append(self.new_topics2id[key])

        super().__init__(pretrained_model, self.new_topics, *args, **kwargs)

    def __call__(self, contexts, batch_size=1):
        outputs = super().__call__(contexts, batch_size)
        outputs = np.hstack([np.max(outputs[:, self.mapping[topic]], axis=-1, keepdims=True)
                             for topic in self.target_topics])
        outputs = np_softmax(outputs)

        return outputs


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage:\tpython3 get_topics.py topics.txt input_file.txt\n\tpython3 get_topics.py topics.txt < '
              'input_file.txt')
        exit(1)

    with open(sys.argv[1], 'rt') as f:
        topics = [topic.rstrip().replace('_', ' ') for topic in f]

    input_stream = open(sys.argv[2], 'rt') if len(sys.argv) == 3 else sys.stdin

    clf = NLITopicClassifier('roberta-large-mnli', topics=topics)

    for line in input_stream:
        line = line.rstrip()
        output_probs = clf(line)[0]
        topic_dist = sorted(list(zip(output_probs, topics)), reverse=True)
        print(line)
        pprint(topic_dist)
        print()
