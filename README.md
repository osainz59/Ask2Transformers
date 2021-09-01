<h1 align="center">Ask2Transformers</h1>
<h3 align="center">A Framework for Entailment based Zero Shot text classification</h3>
<p align="center">
 <a href="https://paperswithcode.com/sota/domain-labelling-on-babeldomains?p=ask2transformers-zero-shot-domain-labelling">
  <img align="center" alt="Contributor Covenant" src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ask2transformers-zero-shot-domain-labelling/domain-labelling-on-babeldomains">
 </a>
</p>

This repository contains the code for out of the box ready to use zero-shot classifiers among different tasks, such as Topic Labelling or Relation Extraction. It is built on top of 🤗 HuggingFace [Transformers](https://github.com/huggingface/transformers) library, so you are free to choose among hundreds of models. You can either, use a dataset specific classifier or define one yourself with just labels descriptions or templates! The repository contains the code for the following publications:

- 📄 [Ask2Transformers - Zero Shot Domain Labelling with Pretrained Transformers](https://arxiv.org/abs/2101.02661) accepted in [GWC2021](http://globalwordnet.org/global-wordnet-conferences-2/).
- 📄 **(Coming soon)** [Label Verbalization and Entailment for Effective Zero- and Few-Shot Relation Extraction]() accepted in [EMNLP2021](https://2021.emnlp.org/)

### Supported (and benchmarked) tasks:
Follow the links to see some examples of how to use the library on each task.
- [Topic classification](./a2t/topic_classification/) evaluated on BabelDomains (Camacho-
Collados and Navigli, 2017)  dataset.
- [Relation classification](./a2t/relation_classification/) evaluated on TACRED (Zhang et al., 2017) dataset.


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

[//]: <img src="./imgs/RE_NLI.svg" style="background-color: white; border-radius: 15px">


## Available models
By default, `roberta-large-mnli` checkpoint is used to perform the inference. You can try different models to perform the zero-shot classification, but they need to be finetuned on a NLI task and be compatible with the `AutoModelForSequenceClassification` class from Transformers. For example:

* `roberta-large-mnli`
* `joeddav/xlm-roberta-large-xnli`
* `facebook/bart-large-mnli`
* `microsoft/deberta-v2-xlarge-mnli` 

**Coming soon:** `t5-large` like generative models support.

## Citation
Cite this paper if you want to cite stuff related to Relation Extraction, etc.
```bibtex
Coming soon.
``` 

Cite this paper if you want to cite stuff related with the library or topic labelling (A2TDomains or our paper results).
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