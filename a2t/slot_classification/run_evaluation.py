import argparse
import json
import os
from pprint import pprint
from collections import Counter
import torch

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

from .data import SlotFeatures
from .wikievents import WikiEventsArgumentDataset
from .mnli import NLISlotClassifierWithMappingHead
from a2t.slot_classification.utils import apply_threshold, find_optimal_threshold, apply_individual_threshold, find_optimal_individual_threshold

CLASSIFIERS = {
    'mnli-mapping': NLISlotClassifierWithMappingHead
}

parser = argparse.ArgumentParser(prog='run_evaluation', description="Run a evaluation for each configuration.")
parser.add_argument('--config', type=str, dest='config', default='experiments/slot_classification/config.json',
                    help='Configuration file for the experiment.')

args = parser.parse_args()

with open(args.config, 'rt') as f:
    config = json.load(f)

for configuration in config:
    # Create the output folder
    os.makedirs(f"experiments/{configuration['name']}", exist_ok=True)

    # Generate the label mappings
    label2id = {label:i for i, label in enumerate(configuration['labels'])}
    n_labels = len(configuration['labels'])

    # Load the datasets
    dev_dataset = WikiEventsArgumentDataset(
        configuration['dev_file'], 
        create_negatives=True,
        max_sentence_distance=configuration.get('max_sentence_distance', None),
        mark_trigger=configuration.get('mark_trigger', False)
    )
    dev_labels = np.array([ label2id[inst.role] for inst in dev_dataset ])
    if 'test_file' in configuration:
        test_dataset = WikiEventsArgumentDataset(
            configuration['test_file'], 
            create_negatives=True,
            max_sentence_distance=configuration.get('max_sentence_distance', None),
            mark_trigger=configuration.get('mark_trigger', False)
        )
        test_labels = np.array([ label2id[inst.role] for inst in test_dataset ])

    results = {}

    for pretrained_model in configuration['nli_models']:

        results[pretrained_model] = {}
        os.makedirs(f"experiments/{configuration['name']}/{pretrained_model}", exist_ok=True)

        classifier = CLASSIFIERS[configuration['classification_model']](
            pretrained_model=pretrained_model,
            negative_threshold=0.0, 
            **configuration
        )

        # Dev
        output = classifier(
            dev_dataset.instances, 
            batch_size=configuration['batch_size'],
            multiclass=True
        )
        output[dev_labels==label2id["OOR"], 0] = 1.0

        # Save the output
        os.makedirs(f"experiments/{configuration['name']}/{pretrained_model}/dev", exist_ok=True)
        np.save(f"experiments/{configuration['name']}/{pretrained_model}/dev/output.npy", output)
        np.save(f"experiments/{configuration['name']}/{pretrained_model}/dev/labels.npy", dev_labels)

        positive_mask = np.logical_and(dev_labels>0, dev_labels<label2id["OOR"])
        positive_acc = accuracy_score(dev_labels[positive_mask] - 1, output[positive_mask, 1:].argmax(-1))

        # Individual threshold
        optimal_indv_threshold, _ = find_optimal_individual_threshold(dev_labels, output, n_labels=n_labels)
        output_ = apply_individual_threshold(output, optimal_indv_threshold)

        with open(f"experiments/{configuration['name']}/{pretrained_model}/dev/predictions.indv.jsonl", 'wt') as f:
            for inst in dev_dataset.to_dict([configuration['labels'][o] for o in output_]):
                f.write(f"{json.dumps(inst)}\n")

        pre_indv, rec_indv, f1_indv, _ = precision_recall_fscore_support(
            dev_labels, output_, average='micro', labels=list(range(1, n_labels))
        )

        # Global threshold
        optimal_global_threshold, _ = find_optimal_threshold(dev_labels, output)
        output_ = apply_threshold(output, threshold=optimal_global_threshold)

        with open(f"experiments/{configuration['name']}/{pretrained_model}/dev/predictions.global.jsonl", 'wt') as f:
            for inst in dev_dataset.to_dict([configuration['labels'][o] for o in output_]):
                f.write(f"{json.dumps(inst)}\n")

        pre_global, rec_global, f1_global, _ = precision_recall_fscore_support(
            dev_labels, output_, average='micro', labels=list(range(1, n_labels))
        )
        
        results[pretrained_model]['global_threshold'] = optimal_global_threshold
        results[pretrained_model]['dev'] = {
            'pos_accuracy': positive_acc,
            'precision_global': pre_global,
            'recall_global': rec_global,
            'f1-score_global': f1_global,
            'precision_indv': pre_indv,
            'recall_indv': rec_indv,
            'f1-score_indv': f1_indv,
            'OOR%': 100 - 100*positive_mask.sum()/(dev_labels>0).sum()
        }

        with open(f"experiments/{configuration['name']}/results.json", 'wt') as f:
            json.dump(results, f, indent=4)

        if 'test_file' in configuration:
            # Test
            output = classifier(
                test_dataset.instances, 
                batch_size=configuration['batch_size'],
                multiclass=True
            )
            output[test_labels==label2id["OOR"], 0] = 1.0

            # Save the output
            os.makedirs(f"experiments/{configuration['name']}/{pretrained_model}/test", exist_ok=True)
            np.save(f"experiments/{configuration['name']}/{pretrained_model}/test/output.npy", output)
            np.save(f"experiments/{configuration['name']}/{pretrained_model}/test/labels.npy", test_labels)

            positive_mask = np.logical_and(test_labels>0, test_labels<label2id["OOR"])
            positive_acc = accuracy_score(test_labels[positive_mask] - 1, output[positive_mask, 1:].argmax(-1))

            # Individual threshold
            output_ = apply_individual_threshold(output, optimal_indv_threshold)

            with open(f"experiments/{configuration['name']}/{pretrained_model}/test/predictions.indv.jsonl", 'wt') as f:
                for inst in test_dataset.to_dict([configuration['labels'][o] for o in output_]):
                    f.write(f"{json.dumps(inst)}\n")

            pre_indv, rec_indv, f1_indv, _ = precision_recall_fscore_support(
                test_labels, output_, average='micro', labels=list(range(1, n_labels))
            )

            # Global threshold
            output_ = apply_threshold(output, threshold=optimal_global_threshold)

            with open(f"experiments/{configuration['name']}/{pretrained_model}/test/predictions.global.jsonl", 'wt') as f:
                for inst in test_dataset.to_dict([configuration['labels'][o] for o in output_]):
                    f.write(f"{json.dumps(inst)}\n")

            pre_global, rec_global, f1_global, _ = precision_recall_fscore_support(
                test_labels, output_, average='micro', labels=list(range(1, n_labels))
            )
            
            results[pretrained_model]['test'] = {
                'pos_accuracy': positive_acc,
                'precision_global': pre_global,
                'recall_global': rec_global,
                'f1-score_global': f1_global,
                'precision_indv': pre_indv,
                'recall_indv': rec_indv,
                'f1-score_indv': f1_indv,
                'OOR%': 100 - 100*positive_mask.sum()/(test_labels>0).sum()
            }

            with open(f"experiments/{configuration['name']}/results.json", 'wt') as f:
                json.dump(results, f, indent=4)

        classifier.clear_gpu_memory()
        del classifier
        torch.cuda.empty_cache()