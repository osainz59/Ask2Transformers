import os
import sys

import numpy as np
import torch


def np_softmax(x, dim=-1):
    e = np.exp(x)
    return e / np.sum(e, axis=dim, keepdims=True)


class TopicClassifier(object):
    def __init__(self, pretrained_model, topics, use_cuda=True, half=False, verbose=True):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.topics = topics
        self.use_cuda = use_cuda
        self.half = half
        self.verbose = verbose

        # Supress stdout printing for model downloads
        if not verbose:
            sys.stdout = open(os.devnull, "w")
            self._initialize(pretrained_model)
            sys.stdout = sys.__stdout__
        else:
            self._initialize(pretrained_model)

        self.model.to(self.device)
        self.model.eval()
        if self.use_cuda and self.half and torch.cuda.is_available():
            self.model.half()

    def _initialize(self, pretrained_model):
        raise NotImplementedError

    def __call__(self, context, batch_size=1):
        raise NotImplementedError
