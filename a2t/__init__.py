"""This repository contains the code for out of the box ready to use zero-shot classifiers among different tasks, 
such as Topic Labelling or Relation Extraction. It is built on top of 🤗 HuggingFace [Transformers](https://github.com/huggingface/transformers) 
library, so you are free to choose among hundreds of models. You can either, use a dataset specific classifier or define one 
yourself with just labels descriptions or templates!

## Installation

By using Pip (check the last release)

```shell
pip install a2t
```

Or by clonning the repository from 
<img 
    src="https://raw.githubusercontent.com/gilbarbara/logos/master/logos/github-icon.svg" 
    width="25" height="25" href="https://github.com/osainz59/Ask2Transformers" /> 
[GitHub](https://github.com/osainz59/Ask2Transformers):

```shell
git clone https://github.com/osainz59/Ask2Transformers.git
cd Ask2Transformers
python -m pip install .
```

## Getting Started

The framework is organized to differentiate **three** main components: the **data**, **task** and **inference**. 
Let's define a Topic Classifier that will classify sentences into the following topics: Politics, Culture, Economy, 
Biology, Legal, Medicine and Business.

### Defining the dataset

We will create a dummy dataset with only one instance to test our model. The `Dataset` object is intended to be 
used to load some data from a file and create the task features.

```python
from a2t.data import Dataset
from a2t.tasks import TopicClassificationFeatures

labels = [
    'politics', 'culture', 'economy', 'biology', 'legal', 'medicine', 'business'
]

class DummyTopicClassificationDataset(Dataset):
    def __init__(self) -> None:
        super().__init__(labels=labels)

        self.append(
            TopicClassificationFeatures(
                context="hospital: a health facility where patients"
                        " receive treatment.",
                label="medicine"
            )
        )

dataset = DummyTopicClassificationDataset()
```

You do not actually need to define a dataset, just a list of `Features` is enough.

### Defining the Task

The `Task` object will contain the **label verbalizations** and other task specific information. In this case it will be just the 
labels that we defined before. For more complex tasks like Relation Extraction you will probably need to define a set of `templates` 
and `valid_conditions`. This object should contain all the information regarding the task like the schema or ontology.

```python
from a2t.tasks import TopicClassificationTask

task = TopicClassificationTask(name="DummyTopic task", labels=labels)
```

### Defining the inference

The `EntailmentClassifier` object should be instantiated with the information about the pre-trained model and device information. You can 
make use of any entailment model available on 🤗 [Transformers](https://github.com/huggingface/transformers) that was
trained on some NLI dataset.

```python
from a2t.base import EntailmentClassifier

nlp = EntailmentClassifier(
    'roberta-large-mnli', 
    use_tqdm=False, 
    use_cuda=True, 
    half=True
)
```

### Putting all together

The following code is enough to run the model:

```python
predictions = nlp(
    task=task, 
    features=dataset, 
    return_labels=True, 
    return_confidences=True, 
    topk=3
)

print(predictions)
```
The result should be something close to this:
```python
[
    [('medicine', 0.8545), ('biology', 0.03693), ('business', 0.0322)]
]
```

## Information Extraction with Entailment

On the previous example we already saw how to create a Topic Classifier that will classify the whole given text into a set of topic labels.
That kind of tasks are known as **Text Classification** tasks. On Information Extraction (IE) instead, we usually find tasks that require 
to classify spans inside a sentence (**Span Classification** tasks like NER) or relations between spans inside a sentence/document (**Tuple 
Classification tasks** like Relation Extraction). This framework differentiates the task types by the number of spans involved: if no spans 
are involved are `ZeroaryTask`, if 1 span is involved are `UnaryTask` and if 2 spans are involved the tasks are `BinaryTask`.

Let's build an small Relation Classifier based on [Sainz et al. (2021)](https://aclanthology.org/2021.emnlp-main.92/):

### Defining the templates for the task

We are going to build a small classifier that will classify entity pairs into the next relations:

* `per:city_of_death`: The `X` entity died in `Y` and `Y` is a city.
* `org:founded_by`: The `X` organization was founded by `Y` person.
* `no_relation`: No relation (among the predefined relations) exists between `X` and `Y`.

```python
labels = ["no_relation", "per:city_of_death", "org:founded_by"]
```

Then, we need to define the set of templates that verbalize each relation. These templates must contain **placeholders** for
two entity spans. In this case, we are going to use `X` and `Y` as they are already defined on `BinaryFeatures`.

```python
templates = {
    "per:city_of_death": [
        "{X} died in {Y}"
    ],
    "org:founded_by": [
        "{X} was founded by {Y}",
        "{Y} founded {X}"
    ]
}
```
We can also define a set of constraints to each of the labels to simplify the templates. For example, on [TACRED](https://nlp.stanford.edu/projects/tacred/)
exists both `per:city_of_death` and `per:country_of_death`, we could use the template `"{X} died in {Y}"` for both and differentiate them adding the 
`"PERSON:CITY"` constraint to `per:city_of_death` as follows:

```python
valid_conditions = {
    "per:city_of_death": [
        "PERSON:CITY",
        "PERSON:LOCATION"
    ],
    "org:founded_by": [
        "ORGANIZATION:PERSON"
    ]
}
```

Once we defined our **labels**, **templates** and **constraints** we can define our task as follows:

```python
from a2t.tasks import BinaryTask, BinaryFeatures

task = BinaryTask(
    name="Relation Classification task",
    required_variables=["X", "Y"],
    additional_variables=["inst_type"],
    labels=labels,
    templates=templates,
    valid_conditions=valid_conditions,
    negative_label_id=0,
    multi_label=True,
    features_class=BinaryFeatures
)
```

### Testing the Relation Classifier

At this point we have all we need to perform inferences on this task, let's see how it actually works:

```python
from a2t.base import EntailmentClassifier

nlp = EntailmentClassifier(
    "microsoft/deberta-v2-xlarge-mnli",
    use_tqdm=False,
    use_cuda=True, 
    half=True
)

test_examples = [
    BinaryFeatures(X='Billy Mays', Y='Tampa', inst_type='PERSON:CITY', context='Billy Mays, the bearded, boisterous pitchman who, as the undisputed king of TV yell and sell, became an unlikely pop culture icon, died at his home in Tampa, Fla, on Sunday', label='per:city_of_death'),
    BinaryFeatures(X='Old Lane Partners', Y='Pandit', inst_type='ORGANIZATION:PERSON', context='Pandit worked at the brokerage Morgan Stanley for about 11 years until 2005, when he and some Morgan Stanley colleagues quit and later founded the hedge fund Old Lane Partners.', label='org:founded_by'),
    BinaryFeatures(X='He', Y='University of Maryland in College Park', inst_type='PERSON:ORGANIZATION', context='He received an undergraduate degree from Morgan State University in 1950 and applied for admission to graduate school at the University of Maryland in College Park.', label='no_relation')
]

nlp(task=task, features=test_examples, return_labels=True, return_confidences=True)
```

The output should look like:

```python
[('per:city_of_death', 0.98828125),
 ('org:founded_by', 0.955078125),
 ('no_relation', 1.0)]
```

For more information consider reading the [Tasks](tasks/index.html) documentation. 
"""

__version__ = "0.4.0"

__pdoc__ = {
    "legacy": False,
    "evaluation": True,
    "tests": False,
    "tasks.base": False,
    "tasks.span_classification": False,
    "tasks.text_classification": False,
    "tasks.tuple_classification": False,
    "base.np_softmax": False,
    "base.np_sigmoid": False,
    "utils": False,
    "base.Classifier": False,
}
