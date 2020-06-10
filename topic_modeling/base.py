import sys, os
import torch

class TopicCLassifier(object):

    def __init__(self, pretrained_model, topics, use_cuda=True):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.topics = topics

        # Supress stdout printing for model downloads
        sys.stdout = open(os.devnull, 'w')
        self._initialize(pretrained_model)
        sys.stdout = sys.__stdout__
        
    def _initialize(self, pretrained_model):
        raise NotImplementedError

    def __call__(self, context, batch_size=1):
        raise NotImplementedError