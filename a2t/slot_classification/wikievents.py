from typing import List
from pprint import pprint
from copy import deepcopy

import numpy as np
import torch
import json
from tqdm import tqdm
#from nltk.tokenize import sent_tokenize

from collections import Counter, defaultdict

from a2t.relation_classification.mnli import NLIRelationClassifierWithMappingHead
from .data import SlotFeatures

def sent_tokenize(text, start_pos):
    """TODO
    """
    import spacy
    nlp = spacy.load("en_core_web_sm")
    for sent in nlp(text).sents:
        yield [sent[0].idx + start_pos, sent.text]

def find_subsentence(offset, sentences):
    sentences_ = sentences.copy() + [(sentences[-1][0] + len(sentences[-1][1]) + 1, "")]
    # if len(sentences) == 1:
    #     return 0 if offset >= sentences[0][0] and offset < sentences[0][0] + len(sentences[0][1]) else -1
    return next((i-1 for i, (idx, sent) in enumerate(sentences_) if offset < idx), -1)

# def get_cluster_id(entity_id, clusters):
#     return next((i for i, c in enumerate(clusters) if entity_id in c), None)

# def create_sent_from_tokens(tokens, start_idx=None, end_idx=None):
#     if not start_idx:
#         start_idx = tokens[0][1]
#     if not end_idx:
#         end_idx = tokens[-1][-1]
#     sent = ""
#     for token, start, end in tokens:
#         if start < start_idx:
#             continue
#         sent += token if len(sent) + start_idx == start else " " + token
#         if len(sent) >= end_idx:
#             return sent.rstrip()
#     return sent.rstrip()

# def create_windowed_sentence(tokens, trigger_tokens, window):
#     positions = []
#     for trigger in trigger_tokens:
#         trigger_pos = next( (i for i, token in enumerate(tokens) if token[0] == trigger), None)
#         if trigger_pos is None:
#             raise IndexError(f"Trigger {trigger} not in {tokens}\n{trigger}")
#         positions.append(trigger_pos)
#     trigger_pos = positions[len(positions)//2]
#     sent = create_sent_from_tokens(tokens[max(0, trigger_pos-window):trigger_pos+window])

#     min_pos, max_pos = tokens[max(0, trigger_pos-window)][1], tokens[min(trigger_pos+window, len(tokens)-1)][-1]
#     return sent, min_pos, max_pos


class WikiEventsArgumentDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, filter_events=None, 
                create_negatives=True, max_sentence_distance=3,
                mark_trigger=False, force_preprocess=False, **kwargs):
        super().__init__()

        self.data_path = data_path
        self.max_sentence_distance = max_sentence_distance
        self.create_negatives = create_negatives
        self.filter_events = filter_events
        self.mark_trigger = mark_trigger

        path_name = data_path.replace('.jsonl', '')
        path_name = f"{path_name}.prepro.{max_sentence_distance}.{create_negatives}.jsonl"
        if self.mark_trigger:
            path_name = path_name.replace('.jsonl', '.trigger.jsonl')

        try:
            if force_preprocess:
                raise Exception()
            self._from_preprocessed(path_name)
        except:
            self._load()
            self._save_preprocessed(path_name)
        
        print(path_name)
        
        self.labels = list(set(inst.role for inst in self.instances))
        self.label2id = {label:i for i, label in enumerate(self.labels)}
        self.id2label = self.labels.copy()
            

    def __getitem__(self, idx):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def _save_preprocessed(self, path):
        with open(path, 'wt') as f:
            for instance in self.instances:
                f.write(f"{json.dumps(instance.__dict__)}\n")
    
    def _from_preprocessed(self, path):
        with open(path) as f:
            self.instances = [
                SlotFeatures(**json.loads(line))  for line in f
            ]

    def _load(self):
        self.instances = []
        with open(self.data_path) as data_f:
            for i, data_line in tqdm(enumerate(data_f)):
                instance = json.loads(data_line)
                entities = {entity['id']: entity for entity in instance['entity_mentions']}

                tokens = [
                    token for sentence in instance['sentences'] for token in sentence[0]
                ]
                
                if self.max_sentence_distance != None:
                    all_sub_sentences = [
                        list(sent_tokenize(text, tokens[0][1])) for tokens, text in instance['sentences']
                    ]

                for event in instance['event_mentions']:

                    if self.filter_events: 
                        event_types = event['event_type'].split('.')
                        if all([
                            event_types[0] not in self.filter_events, 
                            ".".join(event_types[:2]) not in self.filter_events, 
                            ".".join(event_types) not in self.filter_events
                        ]):
                            continue

                    sentence = instance['sentences'][event['trigger']['sent_idx']][-1]
                    sent_entities = {key: deepcopy(entity) for key, entity in entities.items() 
                                        if entity['sent_idx'] == event['trigger']['sent_idx']}
                    # sub_sentences = sent_tokenize(sentence, instance['sentences'][event['trigger']['sent_idx']][0][0][1])
                    # sub_sentences = list(sub_sentences)
                    if self.max_sentence_distance != None:
                        sub_sentences = deepcopy(all_sub_sentences[event['trigger']['sent_idx']])
                        trigger_sub_sentence = find_subsentence(tokens[event['trigger']['start']][1], sub_sentences)

                        if self.mark_trigger:
                            start_pos, trigger_sub_sentence_ = sub_sentences[trigger_sub_sentence]

                            marked_sentence = trigger_sub_sentence_[:tokens[event['trigger']['start']][1]-start_pos] + '<trg> ' + \
                                            trigger_sub_sentence_[tokens[event['trigger']['start']][1]-start_pos:tokens[event['trigger']['end']-1][-1]-start_pos] + ' <trg>' + \
                                            trigger_sub_sentence_[tokens[event['trigger']['end']-1][-1]-start_pos:]
                            
                            sub_sentences[trigger_sub_sentence][-1] = marked_sentence

                        if trigger_sub_sentence < 0:
                            pprint(event['trigger'])
                            pprint(tokens[event['trigger']['start']])
                            print(sentence)
                            pprint(sub_sentences)
                            raise ValueError("Trigger sub-sentence idx must be greater than 0. Found " + str(trigger_sub_sentence))

                    for argument in event['arguments']:

                        label = argument['role'] if entities[argument['entity_id']]['sent_idx'] == event['trigger']['sent_idx'] else 'OOR'

                        if self.max_sentence_distance != None:
                            arg_sub_sentence = find_subsentence(
                                tokens[entities[argument['entity_id']]['start']][1], sub_sentences
                            )

                            if (abs(trigger_sub_sentence - arg_sub_sentence) <= self.max_sentence_distance) and arg_sub_sentence >= 0 \
                                and entities[argument['entity_id']]['sent_idx'] == event['trigger']['sent_idx']:
                                sentence = " ".join([text for _, text in sub_sentences[trigger_sub_sentence:max(
                                            arg_sub_sentence, trigger_sub_sentence+1
                                            )]]) \
                                            if trigger_sub_sentence <= arg_sub_sentence else \
                                            " ".join([text for _, text in sub_sentences[arg_sub_sentence:max(
                                                trigger_sub_sentence, arg_sub_sentence+1
                                            )]])
                            else:
                                sentence = sub_sentences[trigger_sub_sentence][-1]

                            if sentence == "":
                                print(trigger_sub_sentence, arg_sub_sentence, entities[argument['entity_id']]['sent_idx'], event['trigger']['sent_idx'])
                                print(sub_sentences[trigger_sub_sentence:arg_sub_sentence])
                                raise ValueError()

                            
                            label = label if abs(trigger_sub_sentence - arg_sub_sentence) <= self.max_sentence_distance else 'OOR'
                        
                        self.instances.append(
                            SlotFeatures(
                                docid=instance['doc_id'],
                                trigger=event['trigger']['text'],
                                trigger_id=event['id'],
                                trigger_type=event['event_type'],
                                trigger_sent_idx=event['trigger']['sent_idx'],
                                arg=argument['text'],
                                arg_id=argument['entity_id'],
                                arg_type=entities[argument['entity_id']]['entity_type'],
                                arg_sent_idx=entities[argument['entity_id']]['sent_idx'],
                                role=label,
                                pair_type=f"{event['event_type']}:{entities[argument['entity_id']]['entity_type']}",
                                context=sentence
                            )
                        )

                        if argument['entity_id'] in sent_entities:
                            sent_entities.pop(argument['entity_id'])

                    if self.create_negatives:
                        for key, entity in sent_entities.items():

                            if entity['sent_idx'] != event['trigger']['sent_idx']:
                                continue

                            if self.max_sentence_distance != None:
                                arg_sub_sentence = find_subsentence(
                                    tokens[entity['start']][1], sub_sentences
                                )

                                if abs(trigger_sub_sentence - arg_sub_sentence) > self.max_sentence_distance:
                                    continue
                                
                                sentence = " ".join([text for _, text in sub_sentences[trigger_sub_sentence:max(
                                        arg_sub_sentence, trigger_sub_sentence+1
                                        )]]) \
                                        if trigger_sub_sentence <= arg_sub_sentence else \
                                        " ".join([text for _, text in sub_sentences[arg_sub_sentence:max(
                                            trigger_sub_sentence, arg_sub_sentence+1
                                        )]])

                            self.instances.append(
                                SlotFeatures(
                                    docid=instance['doc_id'],
                                    trigger=event['trigger']['text'],
                                    trigger_id=event['id'],
                                    trigger_type=event['event_type'],
                                    trigger_sent_idx=event['trigger']['sent_idx'],
                                    arg=entity['text'],
                                    arg_id=key,
                                    arg_type=entity['entity_type'],
                                    arg_sent_idx=entity['sent_idx'],
                                    role='no_relation',
                                    pair_type=f"{event['event_type']}:{entities[key]['entity_type']}",
                                    context=sentence
                                )
                            )


    def to_dict(self, predictions):
        instances_copy = deepcopy(self.instances)
        inst_per_doc = defaultdict(list)
        for inst, pred in zip(instances_copy, predictions):
            inst.prediction = pred
            inst_per_doc[inst.docid].append(inst)

        with open(self.data_path) as f:
            for line in f:
                instance = json.loads(line)
                for event in instance['event_mentions']:
                    event['arguments'] = []
                    for pred in inst_per_doc[instance['doc_id']]:
                        if pred.trigger_id == event['id'] and pred.prediction not in ['no_relation', 'OOR']:
                            event['arguments'].append(
                                {
                                    'entity_id': pred.arg_id,
                                    'role': pred.prediction,
                                    'text': pred.arg
                                }
                            )
                yield instance
        



if __name__ == "__main__":
    dataset = WikiEventsArgumentDataset(
        'data/wikievents/test.jsonl', max_sentence_distance=0,
        create_negatives=True, force_preprocess=True, mark_trigger=True
    )
    #pprint(dataset[0])
    pprint(len(dataset))
    #pprint(next((inst for inst in dataset if inst.role == "Target"), None))
    #for i, feature in tqdm(enumerate(dataset)):
    #    context = feature.context
    #pprint(Counter([(inst.role, inst.trigger_type, inst.arg_type) for inst in dataset if inst.role == 'Target']))
    #pprint(dataset[0])
    # pprint(next(dataset.to_dict(['no_relation']*len(dataset))))

    # Count OOR
    counter = Counter([inst.role for inst in dataset])
    positives = sum([value for key, value in counter.items() if key != 'no_relation'])
    print(f"OOR%: {counter['OOR']}/{positives} ({counter['OOR']/positives})")

    # with open('dev.test_conflict.jsonl', 'wt', encoding='utf-8') as f:
    #     for inst in dataset.to_dict(
    #         [inst.role for inst in dataset]
    #     ):
    #         f.write(f"{json.dumps(inst)}\n")

