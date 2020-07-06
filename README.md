# Ask2Transformers - Zero Shot Topic Classification with Pretrained Transformers

Work in progress.

This library contains the code for the Ask2Transformers project.


## Topic classification just with non task specific pretrained models

```python
>>> from topic_classification import NLITopicClassifier
>>> topics = ['politics', 'culture', 'economy', 'biology', 'legal', 'medicine', 'business']
>>> context = "hospital: a health facility where patients receive treatment."

>>> clf = NLITopicClassifier('roberta-large-mnli', topics)

>>> predictions = clf(context)[0]
>>> print(sorted(list(zip(predictions, topics))), reverse=True)

[(0.32655442, 'medicine'),
 (0.19446225, 'biology'),
 (0.11796309, 'politics'),
 (0.11180526, 'culture'),
 (0.10732978, 'economy'),
 (0.07613304, 'business'),
 (0.0657522, 'legal')]

```

## WordNet Dataset (BabelNet Domains)

- 1540 annotated glosses
- 34 domains (classes)

Results (Micro-average):
| Method | Precision | Recall | F1-Score |
|:------:|:---------:|:------:|:--------:|
| Distributional (Camacho-Collados et al. 2016) | **84.0** | 59.8 | 69.9 |
| BabelDomains (Camacho-Collados et al. 2017)   | 81.7 | 68.7 | 74.6 |
| | | | |
| Ask2Transformers | 78.44 | **78.44** | **78.44** |


### Approach evaluation

Next table shows the weighted averaged Precision, Recall and F1-Score along with Top-1, Top-3 and Top-5 Accuracy of each of the implemented approaches.

| Method | Precision | Recall | F1-Score | Top-1 | Top-3 | Top-5 |
|:------:|:---------:|:------:|:--------:|:-----:|:-----:|:-----:|
| MNLI (roberta-large-mnli) | **91.6** | **78.44** | **82.4** | **78.44** | **87.46** | **89.74** |
| MNLI (bart-large-mnli) | 85.63 | 61.81 | 66.38 | 61.81 | 79.85 | 87.59 |
| NSP (bert-large-uncased) | 49.78 | 2.07 | 2.83 | 2.07 | 8.57 | 16.49 |
| NSP (bert-base-uncased) | 18.59 | 2.85 | 1.84 | 2.85 | 10.32 | 16.88 |
| MLM (roberta-large) | 71.21 | 12.92 | 16.24 | 12.91 | 30.9 | 45.84 |
| MLM (roberta-base)  | 67.74 | 23.7 | 32.35 | 23.7 | 46.23 | 62.53 |

![Top-K Accuracy curve](/experiments/topk_accuracy_curve.png)
