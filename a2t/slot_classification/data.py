
from dataclasses import dataclass
from typing import List

@dataclass
class SlotFeatures:
    docid: str
    trigger: str
    trigger_id: str
    trigger_type: str # Event type
    trigger_sent_idx: int
    arg: str
    arg_id: str
    arg_type: str # NER
    arg_sent_idx: int
    role: str
    pair_type: str
    context: str
    prediction: str = None