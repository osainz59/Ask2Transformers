import sys
from collections import defaultdict
from pprint import pprint
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from a2t.base import Classifier, np_softmax


class _NLITopicClassifier(Classifier):
    def __init__(
        self,
        labels: List[str],
        *args,
        pretrained_model: str = "roberta-large-mnli",
        use_cuda=True,
        query_phrase="The domain of the sentence is about",
        entailment_position=2,
        half=False,
        verbose=True,
        **kwargs,
    ):
        super().__init__(
            labels,
            pretrained_model=pretrained_model,
            use_cuda=use_cuda,
            verbose=verbose,
            half=half,
        )
        self.query_phrase = query_phrase
        self.ent_pos = entailment_position

        def idx2topic(idx):
            return self.labels[idx]

        self.idx2topic = np.vectorize(idx2topic)

    def _initialize(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)

    def _run_batch(self, batch):
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(batch, padding=True)
            input_ids = torch.tensor(input_ids["input_ids"]).to(self.device)
            output = self.model(input_ids)[0][:, self.ent_pos].view(input_ids.shape[0] // len(self.labels), -1)
            output = output.detach().cpu().numpy()

        return output

    def __call__(self, contexts: List[str], batch_size: int = 1):
        if not isinstance(contexts, list):
            contexts = [contexts]

        batch, outputs = [], []
        for i, context in tqdm(enumerate(contexts), total=len(contexts)):
            sentences = [f"{context} {self.tokenizer.sep_token} {self.query_phrase} {topic}." for topic in self.labels]
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
            topics = self.idx2topic(topics)
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


class NLITopicClassifier(_NLITopicClassifier):
    """NLITopicClassifier

    Zero-Shot topic classifier based on a Natural Language Inference pretrained Transformer.

    Use:

    ```python
    >>> from a2t.topic_classification import NLITopicClassifier

    >>> topics = ['politics', 'culture', 'economy', 'biology', 'legal', 'medicine', 'business']
    >>> context = "hospital: a health facility where patients receive treatment."

    >>> clf = NLITopicClassifier(topics)

    >>> predictions = clf(context)[0]
    >>> print(sorted(list(zip(predictions, topics)), reverse=True))

    [(0.77885467, 'medicine'),
     (0.08395168, 'biology'),
     (0.040319894, 'business'),
     (0.027866213, 'economy'),
     (0.02357693, 'politics'),
     (0.023382403, 'legal'),
     (0.02204825, 'culture')]

    ```
    """

    def __init__(
        self,
        labels: List[str],
        *args,
        pretrained_model: str = "roberta-large-mnli",
        **kwargs,
    ):
        super(NLITopicClassifier, self).__init__(labels, *args, pretrained_model=pretrained_model, **kwargs)

    def __call__(self, contexts: List[str], batch_size: int = 1):
        outputs = super().__call__(contexts=contexts, batch_size=batch_size)
        outputs = np_softmax(outputs)

        return outputs


class NLITopicClassifierWithMappingHead(_NLITopicClassifier):
    def __init__(
        self,
        labels: List[str],
        topic_mapping: Dict[str, str],
        pretrained_model: str = "roberta-large-mnli",
        *args,
        **kwargs,
    ):
        self.new_topics = list(topic_mapping.keys())
        self.target_topics = labels
        self.new_topics2id = {t: i for i, t in enumerate(self.new_topics)}
        self.mapping = defaultdict(list)
        for key, value in topic_mapping.items():
            self.mapping[value].append(self.new_topics2id[key])

        super().__init__(self.new_topics, *args, pretrained_model=pretrained_model, **kwargs)

        def idx2topic(idx):
            return self.target_topics[idx]

        self.idx2topic = np.vectorize(idx2topic)

    def __call__(self, contexts, batch_size=1):
        outputs = super().__call__(contexts, batch_size)
        outputs = np.hstack([np.max(outputs[:, self.mapping[topic]], axis=-1, keepdims=True) for topic in self.target_topics])
        outputs = np_softmax(outputs)

        return outputs


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:\tpython3 get_topics.py topics.txt input_file.txt\n\tpython3 get_topics.py topics.txt < " "input_file.txt")
        exit(1)

    with open(sys.argv[1], "rt") as f:
        topics = [topic.rstrip().replace("_", " ") for topic in f]

    input_stream = open(sys.argv[2], "rt") if len(sys.argv) == 3 else sys.stdin

    clf = NLITopicClassifier(labels=topics, pretrained_model="roberta-large-mnli")

    for line in input_stream:
        line = line.rstrip()
        output_probs = clf(line)[0]
        topic_dist = sorted(list(zip(output_probs, topics)), reverse=True)
        print(line)
        pprint(topic_dist)
        print()
