# Ask2Transformers - Zero Shot Topic Classification with Pretrained Transformers

Work in progress.

This library contains the code for the Ask2Transformers project.


### Topic classification just with non task specific pretrained models

```python
>>> from topic_classification import NLITopicClassifier
>>> topics = ['politics', 'culture', 'economy', 'biology', 'legal', 'medicine', 'business']
>>> context = "hospital: a health facility where patients receive treatment."

>>> clf = NLITopicClassifier(topics)

>>> predictions = clf(context)
>>> print(sorted(list(zip(predictions, topics))), reverse=True)

[(0.32655442, 'medicine'),
 (0.19446225, 'biology'),
 (0.11796309, 'politics'),
 (0.11180526, 'culture'),
 (0.10732978, 'economy'),
 (0.07613304, 'business'),
 (0.0657522, 'legal')]

```
