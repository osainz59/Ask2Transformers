# Topic classification just with non task specific pretrained models

This repository contains the code for the work [Ask2Transformers - Zero Shot Domain Labelling with Pretrained Transformers](https://arxiv.org/abs/2101.02661) accepted in [GWC2021](http://globalwordnet.org/global-wordnet-conferences-2/).

The Ask2Transformers work aims to automatically annotate textual data without any supervision. Given a particular set of labels (BabelDomains, WNDomains, ...), the system has to classify the data without previous examples. This work uses the Transformers library and its pretrained LMs. We evaluate the systems on BabelDomains dataset (Camacho-Collados and Navigli, 2017) achieving 92.14% accuracy on domain labelling.

A2T Domains (A2TD) is a resource generated as part of the Ask2Transformers work. It consists of WordNet synsets automatically annotated with domain information, such as BabelDomains labels. You can find the publicly available annotations and pre-trained models  [here](https://adimen.si.ehu.es/web/A2TDomains).

## Custom topic classifier example

You can build easily your topic classifier by passing the topics to the classifier. Also, you can use more than a single verbalization for each of the labels, to do that, you can take advantage of `NLITopicClassifierWithMappingHead` class.

**Important**: [Transformers](https://github.com/huggingface/transformers) library actually supports the same method implemented in `ZeroShotClassificationPipeline`.

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

>>> from a2t.topic_classification import NLITopicClassifierWithMappingHead

>>> topic_mapping = {topic: topic for topic in topics}
>>> topic_mapping['health'] = 'medicine'
>>> topic_mapping['money'] = 'economy'

>>> clf = NLITopicClassifierWithMappingHead(topics, topic_mapping=topic_mapping)

>>> predictions = clf(context)[0]
>>> print(sorted(list(zip(predictions, topics)), reverse=True))

[(0.8878417, 'medicine'), 
 (0.028496291, 'biology'), 
 (0.024785504, 'business'), 
 (0.019505909, 'legal'), 
 (0.014774636, 'culture'), 
 (0.012658543, 'economy'), 
 (0.011937407, 'politics')]
```

## BalbelDomains and WordNet topic classifiers

Other thing you can do is directly use a predifined classifier that has some default labels and label mappings. Also you can use the method `predict` which has a nicer interface.

```python
>>> from a2t.topic_classification import BabelDomainsClassifier, WNDomainsClassifier

>>> bd_clf = BabelDomainsClassifier()
>>> bd_clf.predict(context, topk=5, return_confidences=True)

[[('Health and medicine', 0.49646863),
  ('Geography and places', 0.081898175),
  ('Language and linguistics', 0.036702),
  ('Culture and society', 0.029479379),
  ('Royalty and nobility', 0.026661409)]]

>>> wn_clf = WNDomainsClassifier()
>>> wn_clf.predict(context, topk=5, return_confidences=True)

[[('medicine', 0.2970561),
  ('body care', 0.105063796),
  ('alimentation', 0.07902936),
  ('person', 0.072948165),
  ('quality', 0.053485252)]]

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