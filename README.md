# Ask2Transformers - Zero Shot Topic Classification with Pretrained Transformers

Work in progress.

This library contains the code for the Ask2Transformers project.


## Topic classification just with non task specific pretrained models

```python
>>> from a2t.topic_classification import NLITopicClassifier
>>> topics = ['politics', 'culture', 'economy', 'biology', 'legal', 'medicine', 'business']
>>> context = "hospital: a health facility where patients receive treatment."

>>> clf = NLITopicClassifier('roberta-large-mnli', topics)

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

## Instalation

By using Pip (check the last release)

```shell script
pip install a2t
```

Or by clonning the repository

```shell script
git clone https://github.com/osainz59/Ask2Transformers.git
cd Ask2Transformers
python -m pip install .
```

## Evaluation

You can easily evaluate a model with a dataset with the following command. For example to evaluate over the WordNet 
dataset with BabelDomains:

```shell script
python3 -m a2t.topic_classification.run_evaluation \
    data/babeldomains.domain.gloss.tsv \
    data/babel_topics.txt \
    --config path_to_config
```

And the configuration file should be a JSON that looks like:

```json
[
    {
        "name": "mnli_roberta-large-mnli",
        "classification_model": "mnli",
        "pretrained_model": "roberta-large-mnli",
        "query_phrase": "Topic or domain about",
        "batch_size": 1,
        "use_cuda": true,
        "entailment_position": 2,
        ...
    },
    ...
]
```
There are some examples on the `experiments/` directory.


### WordNet Dataset (BabelNet Domains)

- 1540 annotated glosses
- 34 domains (classes)

Results (Micro-average):

| Method | Precision | Recall | F1-Score |
|:------:|:---------:|:------:|:--------:|
| Distributional (Camacho-Collados et al. 2016) | 84.0 | 59.8 | 69.9 |
| BabelDomains (Camacho-Collados et al. 2017)   | 81.7 | 68.7 | 74.6 |
| | | | |
| Ask2Transformers | **92.14** | **92.14** | **92.14** |

