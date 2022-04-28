from copy import deepcopy
from typing import Any, Dict, Iterable, List
import json

try:
    from rich import print
except:
    pass

from .base import Dataset
from a2t.tasks.span_classification import NamedEntityClassificationFeatures
from a2t.tasks.tuple_classification import EventArgumentClassificationFeatures
from a2t.tasks.text_classification import TextClassificationFeatures


class _ACEDataset(Dataset):
    """A class to handle ACE datasets."""

    def __init__(self, labels: List[str], *args, **kwargs) -> None:
        super().__init__(labels, *args, **kwargs)
        self._nlp = None

    def _load(self, input_path: str) -> Iterable[dict]:
        with open(input_path, "rt") as data_f:
            for line in data_f:
                yield json.loads(line.strip())

    def _convert_token_ids_to_char_ids(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        tokens = instance["tokens"]
        prev = 0
        char_ids = []
        for token in tokens:
            l = len(token)
            char_ids.append((prev, prev + l))
            prev += l + 1
        return {**instance, "char_ids": char_ids}


class ACEEventClassificationDataset(_ACEDataset):
    def __init__(self, input_path: str, labels: List[str], *args, **kwargs) -> None:
        """This class converts ACE data files into a list of `a2t.tasks.TextClassificationFeatures`.

        Args:
            input_path (str): The path to the input file.
            labels (List[str]): The possible label set of the dataset.
        """
        super().__init__(input_path, labels, *args, **kwargs)

        for instance in self._load(input_path):
            self.append(
                TextClassificationFeatures(
                    context=instance["sentence"], label=[event["event_type"] for event in instance["event_mentions"]]
                )
            )


class ACEEntityClassificationDataset(_ACEDataset):
    def __init__(self, input_path: str, labels: List[str], *args, **kwargs) -> None:
        """This class converts ACE data files into a list of `a2t.tasks.NamedEntityClassificationFeatures`.

        Args:
            input_path (str): The path to the input file.
            labels (List[str]): The possible label set of the dataset.
        """
        super().__init__(labels, *args, **kwargs)

        if not self._nlp:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")

        for instance in self._load(input_path):
            instance = self._convert_token_ids_to_char_ids(instance)

            text = deepcopy(instance["sentence"])
            chunks = sorted([[chunk.start_char, chunk.end_char, chunk.text, None] for chunk in self._nlp(text).noun_chunks])
            for entity in sorted(instance["entity_mentions"], key=lambda x: x["start"]):
                start = instance["char_ids"][entity["start"]][0]
                end = instance["char_ids"][entity["end"] - 1][1]

                for chunk in chunks:
                    # If entity is inside the chunk
                    if start >= chunk[0] and end <= chunk[1]:
                        chunk[-1] = entity["entity_type"]
                    # Chunk is inside the entity
                    elif chunk[0] >= start and chunk[1] <= entity["end"]:
                        chunk[-1] = entity["entity_type"]
                    # The head of the entity matches the head of the chunk
                    elif chunk[1] == end:
                        chunk[-1] = entity["entity_type"]
                    else:
                        continue

            for chunk in chunks:
                self.append(NamedEntityClassificationFeatures(context=text, label=chunk[-1] if chunk[-1] else "O", X=chunk[2]))


class ACEArgumentClassificationDataset(_ACEDataset):

    label_mapping = {
        "Life:Die|Person": "Victim",
        "Movement:Transport|Place": "Destination",
        "Conflict:Attack|Victim": "Target",
        "Justice:Appeal|Plantiff": "Defendant",
    }

    def __init__(self, input_path: str, labels: List[str], *args, mark_trigger: bool = True, **kwargs) -> None:
        """This class converts ACE data files into a list of `a2t.tasks.EventArgumentClassificationFeatures`.

        Args:
            input_path (str): The path to the input file.
            labels (List[str]): The possible label set of the dataset.
        """
        super().__init__(labels, *args, **kwargs)

        for instance in self._load(input_path):
            tokens = instance["tokens"]
            id2ent = {ent["id"]: ent for ent in instance["entity_mentions"]}
            for event in instance["event_mentions"]:
                event_type = event["event_type"].replace(":", ".").split(".")  # [:-1]
                trigger_type, trigger_subtype = event_type
                event_type = ".".join(event_type)

                entities = {ent["id"] for ent in instance["entity_mentions"]}

                if mark_trigger:
                    context = " ".join(
                        tokens[: event["trigger"]["start"]]
                        + ["[["]
                        + tokens[event["trigger"]["start"] : event["trigger"]["end"]]
                        + ["]]"]
                        + tokens[event["trigger"]["end"] :]
                    )
                else:
                    context = " ".join(tokens)

                for argument in event["arguments"]:
                    # Apply label mapping to sattisfy guidelines constraints
                    role = self.label_mapping.get(f'{event["event_type"]}|{argument["role"]}', argument["role"])

                    # Skip annotation errors
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
                            label=role,
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
