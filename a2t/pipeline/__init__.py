"""The pipelines are callable objects that executes `a2t.tasks.Task` instances sequentially. They are very
useful when the tasks to accomplish need to be executed on a sequential fashion. The Figure 1
shows an example of a Information Extraction pipeline.

<figure>
    <center>
        <img 
        src="https://raw.githubusercontent.com/osainz59/Ask2Transformers/Pipelines/imgs/pipeline.svg" 
        height="80%" width="80%" /> 
        <figcaption align="center"><b>Figure 1:</b> An example of Information Extraction pipeline.</figcaption>
    </center>
</figure>

This module contains the code to implement pipelines composed by `a2t.tasks.Task`s, `a2t.pipeline.filters.Filter`s and `a2t.pipeline.candidates.CandidateGenerator`s.

## Creating a NER + Relation Extraction pipeline

Let's create a simple pipeline that consist in two `a2t.tasks.Task`s, NER and Relation Extraction, a `a2t.pipeline.candidates.CandidateGenerator` to 
generate Relation Extraction candidates from NER output and a `a2t.pipeline.filters.Filter` to remove the negative predictions.

First, we import the dependencies:

```python
from a2t.pipeline import Pipeline 
from a2t.pipeline.candidates import UnaryToBinaryCandidateGenerator
from a2t.pipeline.filters import NegativeInstanceFilter
from a2t.tasks import (
    NamedEntityClassificationTask, 
    NamedEntityClassificationFeatures, 
    RelationClassificationTask,
    RelationClassificationFeatures
)
```

Now we define the schema for NER and RE tasks. We are going to create a NER system that classifies text spans
into Person, Location and Organization labels and a Relation Extraction system that extracts the `per:city_of_death`
and `org:founded_by` relations.
```python
ner_labels = ["O", "Person", "Location", "Organization"]

re_labels = ["no_relation", "per:city_of_death", "org:founded_by"]
re_templates = {
    "per:city_of_death": ["{X} died in {Y}."],
    "org:founded_by": ["{X} was founded by {Y}.", "{Y} founded {X}."],
}
re_valid_conditions = {
    "per:city_of_death": ["Person:Location"],
    "org:founded_by": ["Organization:Person"],
}
```

To define the pipeline, we just need to pass as arguments all the components that composes the pipeline. **Caution**: 
the order matters! Along with the components, we should specify which will be the input and the output. Finally, 
we can specify default values for all tasks such as the `threshold` or the entailment `model`, or use specific values
for each task by adding the name of the task + '_' before the value name. For example: `NER_model`. The input features
should be referenced as `"input_features"` or `None`.

```python
pipe = Pipeline(
    (
        NamedEntityClassificationTask("NER", labels=ner_labels), 
        "input_features", 
        "ner_features"
    ),
    (
        UnaryToBinaryCandidateGenerator(
            NamedEntityClassificationFeatures, 
            RelationClassificationFeatures
        ),
        "ner_features",
        "re_features",
    ),
    (NegativeInstanceFilter("O"), "ner_features", "ner_features"),
    (
        RelationClassificationTask(
            "RE",
            labels=re_labels,
            templates=re_templates,
            valid_conditions=re_valid_conditions,
        ),
        "re_features",
        "re_features",
    ),
    (NegativeInstanceFilter("no_relation"), "re_features", "re_features"),
    threshold=0.5,
    NER_model="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    RE_threshold=0.9,
    use_tqdm=False,
)
```
The pipeline shown above uses a default threshold of `0.5` with a specific threshold for the Relation Extraction
task of `0.9`. Also it uses the `"roberta-large-mnli"` checkpoint as default while for the NER task the 
`"ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"` is used.

Let's define some features to test de pipeline:
```python
features = [
    NamedEntityClassificationFeatures(
        context="Billy Mays, the bearded, boisterous pitchman who, "
        "as the undisputed king of TV yell and sell, became an "
        "unlikely pop culture icon, died at his home in Tampa, Fla, on Sunday",
        label=None,
        inst_type="NNP",
        X="Billy Mays",
    ),
    NamedEntityClassificationFeatures(
        context="Billy Mays, the bearded, boisterous pitchman who, "
        "as the undisputed king of TV yell and sell, became an "
        "unlikely pop culture icon, died at his home in Tampa, Fla, on Sunday",
        label=None,
        inst_type="NNP",
        X="Tampa",
    ),
    NamedEntityClassificationFeatures(
        context="Billy Mays, the bearded, boisterous pitchman who, "
        "as the undisputed king of TV yell and sell, became an "
        "unlikely pop culture icon, died at his home in Tampa, Fla, on Sunday",
        label=None,
        inst_type="NNP",
        X="Sunday",
    ),
]
```
And, to run the pipeline we could simply run:
```python
result = pipe(features)
```
The obtained results should look something similar to:
```python
{
    'ner_features': [
        NamedEntityClassificationFeatures(
            context='Billy Mays, the bearded,' ... ' Fla, on Sunday', 
            label='Person', inst_type='NNP', X='Billy Mays'
        ),
        NamedEntityClassificationFeatures(
            context='Billy Mays, the bearded,' ... ' Fla, on Sunday', 
            label='Location', inst_type='NNP', X='Tampa')
    ],
    're_features': [
        RelationClassificationFeatures(
            context='Billy Mays, the bearded,' ... ' Fla, on Sunday', 
            label='per:city_of_death', 
            inst_type='Person:Location', 
            X='Billy Mays', 
            Y='Tampa')
    ]
}
```
Remember that we did not defined the Date label, and therefore it was
not able to classify `"Sunday"` as a Date.

"""

from .base import Pipeline

__all__ = ["Pipeline"]

__pdoc__ = {
    "base": False,
}
