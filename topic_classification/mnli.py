from base import TopicCLassifier

import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from pprint import pprint
from tqdm import tqdm


class NLITopicClassifier(TopicCLassifier):

    def __init__(self, pretrained_model, topics, *args, use_cuda=True, query_phrase="The domain of the sentence is about",
                entailment_position=2, **kwargs):
        super().__init__(pretrained_model, topics, use_cuda=use_cuda)
        self.query_phrase = query_phrase
        self.ent_pos = entailment_position

    def _initialize(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
        self.model.to(self.device)
        self.model.eval()

    def _run_batch(self, batch):
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(batch, pad_to_max_length=True)
            input_ids = torch.tensor(input_ids['input_ids']).to(self.device)
            output = self.model(input_ids)[0][:,self.ent_pos].view(len(batch) // len(self.topics), -1)
            output = torch.softmax(output, dim=-1).detach().cpu().numpy()
        
        return output

    def __call__(self, contexts, batch_size=1):
        if not isinstance(contexts, list):
            contexts = [contexts]

        batch, outputs = [], []
        for i, context in tqdm(enumerate(contexts), total=len(contexts)):
            sentences = [f"{context} {self.tokenizer.sep_token} {self.query_phrase} \"{topic}\"." for topic in self.topics]
            batch.extend(sentences)

            if (i+1) % batch_size == 0:
                output = self._run_batch(batch)
                outputs.append(output)
                batch = []

        if len(batch) > 0:
            output = self._run_batch(batch)
            outputs.append(output)

        outputs = np.vstack(outputs)

        return outputs

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage:\tpython3 get_topics.py topics.txt input_file.txt\n\tpython3 get_topics.py topics.txt < input_file.txt')
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

