from copy import deepcopy
from pprint import pprint
from typing import Iterable, List
from collections import Counter, defaultdict
import json

from tqdm import tqdm

from .base import Dataset
from a2t.tasks.span_classification import NamedEntityClassificationFeatures

# from a2t.tasks.text_classification import TextClassificationFeatures
from a2t.tasks.tuple_classification import EventArgumentClassificationFeatures


class WikiEventsDataset(Dataset):
    def __init__(self, input_path: str, labels: List[str], *args, **kwargs) -> None:
        super().__init__(labels, *args, **kwargs)
        self._nlp = None

    def _sent_tokenize(self, text, start_pos):
        if not self._nlp:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")

        for sent in self._nlp(text).sents:
            yield [sent[0].idx + start_pos, sent.text]

    @staticmethod
    def _find_subsentence(offset, sentences):
        sentences_ = sentences.copy() + [(sentences[-1][0] + len(sentences[-1][1]) + 1, "")]
        # if len(sentences) == 1:
        #     return 0 if offset >= sentences[0][0] and offset < sentences[0][0] + len(sentences[0][1]) else -1
        return next((i - 1 for i, (idx, sent) in enumerate(sentences_) if offset < idx), -1)

    def _load(self, input_path: str) -> Iterable[dict]:
        """TODO: Finish
        - Convert WikiEvents into a sentence level dataset?
        """
        with open(input_path, "rt") as data_f:
            for data_line in tqdm(data_f):
                instance = json.loads(data_line)

                entities = {entity["id"]: entity for entity in instance["entity_mentions"]}
                tokens = [token for sentence in instance["sentences"] for token in sentence[0]]

                sub_sentence_information = defaultdict(dict)
                all_sub_sentences = [list(self._sent_tokenize(text, tokens[0][1])) for tokens, text in instance["sentences"]]

                for i, sentences in enumerate(all_sub_sentences):
                    for j, sent in enumerate(sentences):
                        sub_sentence_information[f"{i}-{j}"]["text"] = sent[-1]

                for value in entities.values():
                    sent_idx = value["sent_idx"]
                    sub_sent_idx = self._find_subsentence(tokens[value["start"]][1], all_sub_sentences[sent_idx])

                    sentence = sub_sentence_information[f"{sent_idx}-{sub_sent_idx}"]["text"]

                    new_value = deepcopy(value)
                    new_value["start"] = tokens[value["start"]][1] - all_sub_sentences[sent_idx][sub_sent_idx][0]
                    # Fix start if needed
                    _shift = sentence[new_value["start"] :].find(new_value["text"])
                    if _shift == -1:
                        continue
                    elif _shift > 0:
                        new_value["start"] += _shift

                    # new_value["end"] = tokens[value["end"] - 1][-1] - all_sub_sentences[sent_idx][sub_sent_idx][0]
                    new_value["end"] = new_value["start"] + len(new_value["text"])

                    assert (
                        sentence[new_value["start"] : new_value["end"]] == new_value["text"]
                    ), f"{sentence[new_value['start']:new_value['end']]}|{new_value['text']}"

                    if "entity_mentions" not in sub_sentence_information[f"{sent_idx}-{sub_sent_idx}"]:
                        sub_sentence_information[f"{sent_idx}-{sub_sent_idx}"]["entity_mentions"] = []

                    sub_sentence_information[f"{sent_idx}-{sub_sent_idx}"]["entity_mentions"].append(new_value)

                for event in instance["event_mentions"]:
                    sent_idx = event["trigger"]["sent_idx"]
                    sub_sent_idx = self._find_subsentence(tokens[event["trigger"]["start"]][1], all_sub_sentences[sent_idx])

                    sentence = sub_sentence_information[f"{sent_idx}-{sub_sent_idx}"]["text"]

                    new_value = deepcopy(event)
                    new_value["trigger"]["start"] = (
                        tokens[event["trigger"]["start"]][1] - all_sub_sentences[sent_idx][sub_sent_idx][0]
                    )
                    # Fix start if needed
                    _shift = sentence[new_value["trigger"]["start"] :].find(new_value["trigger"]["text"])
                    if _shift == -1:
                        continue
                    elif _shift > 0:
                        new_value["trigger"]["start"] += _shift
                    # new_value["trigger"]["end"] = (
                    #     tokens[event["trigger"]["end"] - 1][-1] - all_sub_sentences[sent_idx][sub_sent_idx][0]
                    # )
                    new_value["trigger"]["end"] = new_value["trigger"]["start"] + len(new_value["trigger"]["text"])

                    sent_entities = (
                        {ent["id"] for ent in sub_sentence_information[f"{sent_idx}-{sub_sent_idx}"]["entity_mentions"]}
                        if "entity_mentions" in sub_sentence_information[f"{sent_idx}-{sub_sent_idx}"]
                        else {}
                    )

                    for argument in new_value["arguments"]:
                        if not argument["entity_id"] in sent_entities:
                            argument["role"] = f"[OOR]_{argument['role']}"

                    assert (
                        sentence[new_value["trigger"]["start"] : new_value["trigger"]["end"]] == new_value["trigger"]["text"]
                    ), f"{sentence[new_value['trigger']['start']:new_value['trigger']['end']]}|{new_value['trigger']['text']}"

                    if "event_mentions" not in sub_sentence_information[f"{sent_idx}-{sub_sent_idx}"]:
                        sub_sentence_information[f"{sent_idx}-{sub_sent_idx}"]["event_mentions"] = []

                    sub_sentence_information[f"{sent_idx}-{sub_sent_idx}"]["event_mentions"].append(new_value)

                for i, sentences in enumerate(all_sub_sentences):
                    for j, _ in enumerate(sentences):
                        info = sub_sentence_information[f"{i}-{j}"]
                        info["doc_id"] = f"{instance['doc_id']}_{i}_{j}"

                        if "entity_mentions" not in info and "event_mentions" not in info and len(info["text"]) <= 25:
                            continue

                        if "entity_mentions" not in info:
                            info["entity_mentions"] = []
                        if "event_mentions" not in info:
                            info["event_mentions"] = []

                        yield info


# class WikiEventsEventExtractionDataset(WikiEventsDataset):
#     def __init__(self, input_path: str, labels: List[str], *args, **kwargs) -> None:
#         super().__init__(input_path, labels, *args, **kwargs)

#         for instance in self._load(input_path):
#             self.append(
#                 TextClassificationFeatures(
#                     context=instance["text"],
#                     label=[".".join(event["event_type"].split(".")[:-1]) for event in instance["event_mentions"]],
#                 )
#             )


class WikiEventsEntityClassificationDataset(WikiEventsDataset):
    def __init__(self, input_path: str, labels: List[str], *args, **kwargs) -> None:
        super().__init__(input_path, labels, *args, **kwargs)

        if not self._nlp:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")

        for instance in self._load(input_path):
            text = deepcopy(instance["text"])
            chunks = sorted([[chunk.start_char, chunk.end_char, chunk.text, None] for chunk in self._nlp(text).noun_chunks])
            for entity in sorted(instance["entity_mentions"], key=lambda x: x["start"]):
                for chunk in chunks:
                    # If entity is inside the chunk
                    if entity["start"] >= chunk[0] and entity["end"] <= chunk[1]:
                        chunk[-1] = entity["entity_type"]
                    # Chunk is inside the entity
                    elif chunk[0] >= entity["start"] and chunk[1] <= entity["end"]:
                        chunk[-1] = entity["entity_type"]
                    # The head of the entity matches the head of the chunk
                    elif chunk[1] == entity["end"]:
                        chunk[-1] = entity["entity_type"]
                    else:
                        continue

            for chunk in chunks:
                self.append(NamedEntityClassificationFeatures(context=text, label=chunk[-1] if chunk[-1] else "O", X=chunk[2]))


class WikiEventsArgumentClassificationDataset(WikiEventsDataset):
    def __init__(self, input_path: str, labels: List[str], *args, mark_trigger: bool = False, **kwargs) -> None:
        super().__init__(input_path, labels, *args, **kwargs)

        for instance in self._load(input_path):

            id2ent = {ent["id"]: ent for ent in instance["entity_mentions"]}
            for event in instance["event_mentions"]:
                event_type = event["event_type"].replace(":", ".").split(".")  # [:-1]
                trigger_type = event_type[0]
                trigger_subtype = event_type[-2]
                event_type = ".".join(event_type)

                entities = {ent["id"] for ent in instance["entity_mentions"]}

                context = instance["text"][:]
                if mark_trigger:
                    context = (
                        context[: event["trigger"]["start"]]
                        + "[["
                        + event["trigger"]["text"]
                        + "]]"
                        + context[event["trigger"]["end"] :]
                    )

                for argument in event["arguments"]:
                    if argument["entity_id"] not in entities:
                        continue

                    self.append(
                        EventArgumentClassificationFeatures(
                            context=context,
                            trg=event["trigger"]["text"],
                            trg_type=trigger_type,
                            trg_subtype=trigger_subtype,
                            inst_type=f"{event_type}:{id2ent[argument['entity_id']]['entity_type']}",
                            arg=id2ent[argument["entity_id"]]["text"],
                            label=argument["role"] if not "OOR" in argument["role"] else "OOR",
                        )
                    )

                    entities.remove(argument["entity_id"])

                # Generate negative examples
                for entity in entities:
                    self.append(
                        EventArgumentClassificationFeatures(
                            context=context,
                            trg=event["trigger"]["text"],
                            trg_type=trigger_type,
                            trg_subtype=trigger_subtype,
                            inst_type=f"{event_type}:{id2ent[entity]['entity_type']}",
                            arg=id2ent[entity]["text"],
                            label="no_relation",
                        )
                    )


if __name__ == "__main__":
    # dataset = WikiEventsDataset("data/wikievents/train.jsonl", [])._load("data/wikievents/train.jsonl")
    # with open(".ignore/wikievents.sentence.train.jsonl", "wt") as f:
    #     for instance in dataset:
    #         f.write(f"{json.dumps(instance)}\n")

    with open("experiments/WikiEvents_arguments/config.json") as f:
        config = json.load(f)

    labels = list()

    dataset = WikiEventsArgumentClassificationDataset(config["dev_path"], config["labels"])
