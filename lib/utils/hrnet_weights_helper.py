"""
Helper functions for converting resnet pretrained weights from other formats
"""
import os
import pickle

import torch

import nn as mynn
import utils.detectron_weight_helper as dwh
from core.config import cfg
from torch import nn


def load_pretrained_imagenet_weights(model):
    """Load pretrained weights
    Args:
        model: the generalized rcnnn module
    """
    pretrained = os.path.join(cfg.ROOT_DIR, cfg.HRNET.IMAGENET_PRETRAINED_WEIGHTS)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    if os.path.isfile(pretrained):
        pretrained_dict = torch.load(pretrained)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Load pre-trained weight for hrnet !!")

    else:
        raise NotImplementedError


