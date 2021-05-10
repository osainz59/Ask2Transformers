# Ask2Transformers - Zero Shot Domain Labelling with Pretrained Transformers

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ask2transformers-zero-shot-domain-labelling/domain-labelling-on-babeldomains)](https://paperswithcode.com/sota/domain-labelling-on-babeldomains?p=ask2transformers-zero-shot-domain-labelling)

This repository contains the code for the work [Ask2Transformers - Zero Shot Domain Labelling with Pretrained Transformers](https://arxiv.org/abs/2101.02661) accepted in [GWC2021](http://globalwordnet.org/global-wordnet-conferences-2/).

The Ask2Transformers work aims to automatically annotate textual data without any supervision. Given a particular set of labels (BabelDomains, WNDomains, ...), the system has to classify the data without previous examples. This work uses the Transformers library and its pretrained LMs. We evaluate the systems on BabelDomains dataset (Camacho-Collados and Navigli, 2017) achieving 92.14% accuracy on domain labelling.

A2T Domains (A2TD) is a resource generated as part of the Ask2Transformers work. It consists of WordNet synsets automatically annotated with domain information, such as BabelDomains labels. You can find the publicly available annotations and pre-trained models  [here](https://adimen.si.ehu.es/web/A2TDomains).

## Topic classification just with non task specific pretrained models

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

## Available models
By default, `roberta-large-mnli` checkpoint is used to perform the inference. You can try different models to perform the zero-shot classification, but they need to be finetuned on a NLI task and be compatible with the `AutoModelForSequenceClassification` class from Transformers. For example:

* `roberta-large-mnli`
* `joeddav/xlm-roberta-large-xnli`
* `facebook/bart-large-mnli`
* `microsoft/deberta-v2-xlarge-mnli` 

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

    },
    
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

## Citation

```bibtex
@inproceedings{sainz-rigau-2021-ask2transformers,
    title = "{A}sk2{T}ransformers: Zero-Shot Domain labelling with Pretrained Language Models",
    author = "Sainz, Oscar  and
      Rigau, German",
    booktitle = "Proceedings of the 11th Global Wordnet Conference",
    month = jan,
    year = "2021",
    address = "University of South Africa (UNISA)",
    publisher = "Global Wordnet Association",
    url = "https://www.aclweb.org/anthology/2021.gwc-1.6",
    pages = "44--52",
    abstract = "In this paper we present a system that exploits different pre-trained Language Models for assigning domain labels to WordNet synsets without any kind of supervision. Furthermore, the system is not restricted to use a particular set of domain labels. We exploit the knowledge encoded within different off-the-shelf pre-trained Language Models and task formulations to infer the domain label of a particular WordNet definition. The proposed zero-shot system achieves a new state-of-the-art on the English dataset used in the evaluation.",
}
```

