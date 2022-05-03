"""The module `tasks` contains the code related to the `Task` definition.

![Task taxonomy.](https://raw.githubusercontent.com/osainz59/Ask2Transformers/master/imgs/task_taxonomy.svg)

The tasks on this module are organized based on the number of spans to classify:

* `ZeroaryTask`: are tasks like Text Classification, where the aim is to classify the given text into a set of predefined
labels.
* `UnaryTask`: are tasks like Named Entity Classification, where the object to classify is an span within a text.
* `BinaryTask`: are tasks like Relation Classification, where what is actually classified is the relation between two spans in a text.

There are also more specific predefined task classes like `TopicClassificationTask` that includes helpful code and default values for 
the given task. You can either create create a task specific class or instantiate one of the predefined ones.

In addition to the `Task` class a `Features` class must be defined. The `Features` class will define which type of information is
used during the classification. For example, for a `UnaryTask` a `context` and some variable `X` for the span are needed. This class
will also be used to instantiate the task data instances.
"""

from .base import Task, ZeroaryTask, UnaryTask, BinaryTask, Features, ZeroaryFeatures, UnaryFeatures, BinaryFeatures
from .text_classification import (
    TopicClassificationFeatures,
    TopicClassificationTask,
    TextClassificationFeatures,
    TextClassificationTask,
)
from .span_classification import NamedEntityClassificationFeatures, NamedEntityClassificationTask
from .tuple_classification import (
    RelationClassificationFeatures,
    RelationClassificationTask,
    EventArgumentClassificationFeatures,
    EventArgumentClassificationTask,
    TACREDRelationClassificationTask,
    TACREDFeatures,
)


PREDEFINED_TASKS = {
    "zero-ary": (ZeroaryTask, ZeroaryFeatures),
    "unary": (UnaryTask, UnaryFeatures),
    "binary": (BinaryTask, BinaryFeatures),
    "topic-classification": (TopicClassificationTask, TopicClassificationFeatures),
    "named-entity-classification": (NamedEntityClassificationTask, NamedEntityClassificationFeatures),
    "relation-classification": (RelationClassificationTask, RelationClassificationFeatures),
    "event-argument-classification": (EventArgumentClassificationTask, EventArgumentClassificationFeatures),
    "tacred": (TACREDRelationClassificationTask, TACREDFeatures),
}


__all__ = [
    "Task",
    "Features",
    "ZeroaryTask",
    "ZeroaryFeatures",
    "UnaryTask",
    "UnaryFeatures",
    "BinaryTask",
    "BinaryFeatures",
    "TopicClassificationFeatures",
    "TopicClassificationTask",
    "TextClassificationFeatures",
    "TextClassificationTask",
    "NamedEntityClassificationFeatures",
    "NamedEntityClassificationTask",
    "RelationClassificationFeatures",
    "RelationClassificationTask",
    "EventArgumentClassificationFeatures",
    "EventArgumentClassificationTask",
    "TACREDFeatures",
    "TACREDRelationClassificationTask",
    "PREDEFINED_TASKS",
]

# Ignore __dataclass_fields__ variables on documentation
__pdoc__ = {
    **{
        f"{_class}.{varname}": False
        for _class in __all__
        if hasattr(eval(_class), "__dataclass_fields__")
        for varname in eval(_class).__dataclass_fields__.keys()
    }
}
