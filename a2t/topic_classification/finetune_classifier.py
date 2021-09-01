from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
from transformers.optimization import get_linear_schedule_with_warmup
import argparse
import json
import gzip
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD, AdamW, Adam

# from apex import amp
from tqdm import tqdm
import numpy as np
from pprint import pprint
import os

try:
    from apex import amp
except:
    pass


class EarlyStopping(object):
    """Simple implementation of Early Stopping concept based on the validation loss.

    Usage:
    ```
    >>> early_stopping = EarlyStopping(patience=3, delta=.001)

    >>> for epoch in epochs:
    >>>     val_loss = ..
    >>>     if early_stopping(val_loss):
    >>>         break
    ```
    """

    def __init__(self, patience=3, delta=0.0, save_checkpoint_fn=None):
        super().__init__()

        self.patience = patience
        self.delta = delta
        self.save_checkpoint_fn = save_checkpoint_fn

        self.best_loss = np.inf
        self.best_epoch = -1
        self.steps = 0
        self.epoch = -1

    def __call__(self, val_loss: float, *args, **kwargs) -> bool:
        self.epoch += 1
        diff = (self.best_loss + self.delta) - val_loss
        if diff > 0:
            self.steps = self.patience
            if diff > self.delta:
                self.best_epoch = self.epoch
                self.best_loss = val_loss
                if self.save_checkpoint_fn:
                    self.save_checkpoint_fn(*args, val_loss=self.best_loss, **kwargs)
        else:
            self.steps -= 1

        return self.steps <= 0

    def get_best_epoch(self):
        return self.best_epoch


def parse_args():

    parser = argparse.ArgumentParser(
        "finetune_classifier",
        usage="finetune_classifier train_data eval_data --config config.json",
    )
    parser.add_argument("--config", type=str, dest="config", help="Training configuration")

    args = parser.parse_args()

    return args


def train(opt):

    # Load the configuration
    with open(opt.config, "rt") as f:
        config = json.load(f)

    os.makedirs(config["output_path"], exist_ok=True)

    # Load the topics/labels
    with open(config["topics"], "rt") as f:
        topics = [topic.rstrip().replace("_", " ") for topic in f]
        topic2id = {topic: i for i, topic in enumerate(topics)}

    # Load the sinset-->gloss mapping
    glosses = {}
    with gzip.open(config["glosses"], "rt") as f:
        for line in f:
            row = line.split()
            idx, gloss = row[0], " ".join(row[1:]).replace("\t", " ")
            glosses[idx] = gloss

    # Load train and eval data
    train_data, train_labels, eval_data, eval_labels = [], [], [], []
    with open(config["eval_data"], "rt") as f:
        for line in f:
            idx, label, _ = line.strip().split("\t")
            eval_data.append(idx)
            eval_labels.append(topic2id[label])

    with gzip.open(config["train_data"], "rt") as f:
        for line in f:
            row = line.strip().split("\t")
            if row[0] in glosses and float(row[-1]) > config["minimum_confidence"] and row[0] not in eval_data:
                train_data.append(row[0])
                train_labels.append(topic2id[row[1]])

    # Convert train and eval data to glosses
    train_glosses = [glosses[instance] for instance in train_data]
    eval_glosses = [glosses[instance] for instance in eval_data]

    # Define the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model"])
    tokenizer.save_pretrained(config["output_path"])
    cfg = AutoConfig.from_pretrained(config["pretrained_model"])
    cfg.num_labels = len(topics)
    cfg.label2id = topic2id
    cfg.id2label = {str(idx): label for label, idx in topic2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(config["pretrained_model"], config=cfg)

    # Prepare data for training
    train_dataset = tokenizer(
        train_glosses,
        padding=True,
        truncation=True,
        max_length=config["max_length"],
    )
    eval_dataset = tokenizer(
        eval_glosses,
        padding=True,
        truncation=True,
        max_length=config["max_length"],
    )

    train_dataset = TensorDataset(
        torch.tensor(train_dataset["input_ids"]),
        torch.tensor(train_dataset["attention_mask"]),
        torch.tensor(train_labels),
    )
    eval_dataset = TensorDataset(
        torch.tensor(eval_dataset["input_ids"]),
        torch.tensor(eval_dataset["attention_mask"]),
        torch.tensor(eval_labels),
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config["eval_batch_size"])

    def get_parameters(model, freeze=False):
        param_optimizer = list(model.named_parameters())
        if freeze:
            for n, p in param_optimizer:
                if "classifier" not in n:
                    p.requires_grad = False
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        return optimizer_grouped_parameters

    optimizer = AdamW(get_parameters(model, config["freeze"]), lr=config["learning_rate"])

    model.cuda()
    if config["half"]:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=True)

    def save_checkpoint_fn(model, output_path, **kwargs):
        model.save_pretrained(output_path)

    config["training_log"] = {}

    early_stopping = EarlyStopping(patience=config["patience"], save_checkpoint_fn=save_checkpoint_fn)
    for epoch in range(config["epochs"]):
        config["training_log"][f"epoch_{epoch}"] = {}
        model.train()
        total_loss, accuracy = 0.0, 0.0
        correct, total = 0, 0
        progress = tqdm(
            enumerate(train_dataloader),
            desc=f"Epoch: {epoch} - Loss: {total_loss} - Accuracy: {accuracy}",
            total=len(train_dataset) // config["batch_size"],
        )
        for i, batch in progress:
            # Batch to cuda
            input_ids, attention_mask, labels = batch

            loss, logits = model(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                labels=labels.cuda(),
            )

            if config["half"]:
                with amp.scale_loss(loss, optimizer) as scale_loss:
                    scale_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            correct += sum(np.argmax(logits.detach().cpu().numpy(), -1) == labels.numpy())
            total += input_ids.shape[0]
            accuracy = correct / total

            progress.set_description(f"Epoch: {epoch} - Loss: {total_loss/(i+1):.3f} - Accuracy: {accuracy:.3f}")

        config["training_log"][f"epoch_{epoch}"]["train_loss"] = total_loss / (i + 1)
        config["training_log"][f"epoch_{epoch}"]["train_accuracy"] = correct / total

        model.eval()
        with torch.no_grad():
            total_loss, accuracy = 0.0, 0.0
            correct, total = 0, 0
            progress = tqdm(
                enumerate(eval_dataloader),
                desc=f"Epoch: {epoch} - Loss: {total_loss} - Accuracy: {accuracy}",
                total=len(eval_dataset) // config["eval_batch_size"],
            )
            for i, batch in progress:
                # Batch to cuda
                input_ids, attention_mask, labels = batch

                loss, logits = model(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    labels=labels.cuda(),
                )

                total_loss += loss.item()
                correct += sum(np.argmax(logits.detach().cpu().numpy(), -1) == labels.numpy())
                total += input_ids.shape[0]
                accuracy = correct / total

                progress.set_description(f"Epoch: {epoch} - Loss: {total_loss/(i+1):.3f} - Accuracy: {accuracy:.3f}")

        config["training_log"][f"epoch_{epoch}"]["eval_loss"] = total_loss / (i + 1)
        config["training_log"][f"epoch_{epoch}"]["eval_accuracy"] = correct / total

        if early_stopping(total_loss / (i + 1), model, output_path=config["output_path"]):
            break

    # Save the new config file
    with open(opt.config, "wt") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    opt = parse_args()
    train(opt)
