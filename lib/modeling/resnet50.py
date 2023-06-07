import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from core.config import cfg
import nn as mynn
import utils.net as net_utils
from torchvision import models

def torch_resnet50():
    return resnet()

class resnet(nn.Module):
    def __init__(self, block_counts=4):
        super(resnet, self).__init__()
        backbone = models.resnet50(pretrained=cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS)
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            print("Load pre-trained weight for hrnet !!")
        else:
            print("Not Load pre-trained weight for hrnet !!")

        self.res1 = nn.Sequential(backbone.conv1,
                                  backbone.bn1,
                                  backbone.relu,
                                  backbone.maxpool)

        self.res2 = backbone.layer1
        self.res3 = backbone.layer2
        self.res4 = backbone.layer3

        if block_counts == 5:
            self.spatial_scale = 1 / 32
            self.dim_out = 2048
            self.res5 = backbone.layer4

            raise AssertionError

        elif block_counts == 4:
            self.spatial_scale = 1 / 16
            self.dim_out = 1024

        else:
            raise AssertionError

        self.block_counts = block_counts

        self._init_modules()

    def _init_modules(self):
        assert cfg.ResNet.FREEZE_AT in [0, 2, 3, 4, 5]
        for i in range(1, cfg.ResNet.FREEZE_AT + 1):
            print("freeze : {}".format('res%d' % i))
            freeze_params(getattr(self, 'res%d' % i))

        # Freeze all bn (affine) layers !!!
        # self.apply(lambda m: freeze_params(m) if isinstance(m, nn.BatchNorm2d) else None)
        self.freeze(self)

    def freeze(self,m):
        for i, k in m.named_children():
            if isinstance(k, nn.BatchNorm2d):
                k.eval()
            else:
                self.freeze(k)

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.ResNet.FREEZE_AT + 1, self.block_counts + 1):
            getattr(self, 'res%d' % i).train(mode)

        self.freeze(self)

    def detectron_weight_mapping(self):
        mapping_to_detectron = {}
        orphan_in_detectron = []
        for name,_ in self.named_parameters():
            mapping_to_detectron[name] = name

        return mapping_to_detectron, orphan_in_detectron

    def forward(self, x):
        for i in range(self.block_counts):
            x = getattr(self, 'res%d' % (i + 1))(x)

        return x


class roi_2mlp_head_refine_model_mask_fuse(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = 4096

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

        self.mask_branch = nn.Sequential(nn.Conv2d(dim_in * 2,dim_in,kernel_size=3,padding=1),
                                         nn.ReLU(),)

        self.seg_fc = nn.Sequential(nn.Linear(dim_in * roi_size**2, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU())

    def detectron_weight_mapping(self):
        detectron_weight_mapping = dict()
        for name, _ in self.named_parameters():
            detectron_weight_mapping[name] = name

        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x, rois, masks):
        box_x = self.roi_xform(
            x, rois,
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        batch_size = box_x.size(0)

        masks = masks.unsqueeze(1)
        mask_x = box_x * masks.expand(-1,1024,-1,-1)

        box_mask_cat = torch.concat((box_x,mask_x),dim=1)
        box_mask_cat = self.mask_branch(box_mask_cat)
        seg_x = self.seg_fc(box_mask_cat.view(batch_size, -1))

        return seg_x

def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
