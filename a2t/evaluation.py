"""Main evaluation script.

Please consider making a copy and customizing it yourself if you aim to use a custom class that is not already 
defined on the library.

### Usage

```bash 
a2t.evaluation [-h] [--config CONFIG]

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  Config with task (schema) and data information.
```

### Configuration file

A configuration file containing the task and evaluation information should look like this:

```json
{
    "name": "BabelDomains",
    "task_name": "topic-classification",
    "features_class": "a2t.tasks.text_classification.TopicClassificationFeatures",
    "hypothesis_template": "The domain of the sentence is about {label}.",
    "nli_models": [
        "roberta-large-mnli"
    ],
    "labels": [
        "Animals",
        "Art, architecture, and archaeology",
        "Biology",
        "Business, economics, and finance",
        "Chemistry and mineralogy",
        "Computing",
        "Culture and society",
        ...
        "Royalty and nobility",
        "Sport and recreation",
        "Textile and clothing",
        "Transport and travel",
        "Warfare and defense"
    ],
    "preprocess_labels": true,
    "dataset": "babeldomains",
    "test_path": "data/babeldomains.domain.gloss.tsv",
    "use_cuda": true,
    "half": true
}
```

"""
import argparse
import json
import os
from pprint import pprint
from types import SimpleNamespace
import numpy as np
import torch

from a2t.tasks import PREDEFINED_TASKS
from a2t.data import PREDEFINED_DATASETS
from a2t.base import EntailmentClassifier


def main(args):

    with open(args.config, "rt") as f:
        config = SimpleNamespace(**json.load(f))

    os.makedirs(f"experiments/{config.name}", exist_ok=True)

    task_class, _ = PREDEFINED_TASKS[config.task_name]
    task = task_class.from_config(args.config)  # (**vars(config))

    dataset_class = PREDEFINED_DATASETS[config.dataset]

    assert hasattr(config, "dev_path") or hasattr(config, "test_path"), "At least a test or dev path must be provided."

    # Run dev evaluation
    if hasattr(config, "dev_path"):
        dev_dataset = dataset_class(config.dev_path, task.labels)
    else:
        dev_dataset = None

    if hasattr(config, "test_path"):
        test_dataset = dataset_class(config.test_path, task.labels)
    else:
        test_dataset = None

    results = {}
    for pretrained_model in config.nli_models:

        nlp = EntailmentClassifier(pretrained_model, **vars(config))

        results[pretrained_model] = {}

        if dev_dataset:
            _, output = nlp(task=task, features=dev_dataset, negative_threshold=0.0, return_raw_output=True, **vars(config))

            dev_labels = dev_dataset.labels

            # Save the output
            os.makedirs(
                f"experiments/{config.name}/{pretrained_model}/dev",
                exist_ok=True,
            )
            np.save(
                f"experiments/{config.name}/{pretrained_model}/dev/output.npy",
                output,
            )
            np.save(
                f"experiments/{config.name}/{pretrained_model}/dev/labels.npy",
                dev_labels,
            )

            # If dev data then optimize the threshold on it
            dev_results = task.compute_metrics(dev_labels, output, threshold="optimize")
            results[pretrained_model]["dev"] = dev_results

            with open(f"experiments/{config.name}/results.json", "wt") as f:
                json.dump(results, f, indent=4)

        if test_dataset:
            _, output = nlp(task=task, features=test_dataset, negative_threshold=0.0, return_raw_output=True, **vars(config))

            test_labels = test_dataset.labels

            # Save the output
            os.makedirs(
                f"experiments/{config.name}/{pretrained_model}/test",
                exist_ok=True,
            )
            np.save(
                f"experiments/{config.name}/{pretrained_model}/test/output.npy",
                output,
            )
            np.save(
                f"experiments/{config.name}/{pretrained_model}/test/labels.npy",
                test_labels,
            )

            optimal_threshold = 0.5 if not dev_dataset else dev_results["optimal_threshold"]
            test_results = task.compute_metrics(test_labels, output, threshold=optimal_threshold)
            results[pretrained_model]["test"] = test_results

            with open(f"experiments/{config.name}/results.json", "wt") as f:
                json.dump(results, f, indent=4)

        nlp.clear_gpu_memory()
        del nlp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("a2t.evaluation")
    parser.add_argument("--config", type=str, help="Config with task (schema) and data information.")

    args = parser.parse_args()
    main(args)
