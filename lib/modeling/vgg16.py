import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from core.config import cfg
import nn as mynn
import utils.net as net_utils

import pynvml
pynvml.nvmlInit()

class FT_VGG(nn.Module):
    def __init__(self, model, num_classes):
        super(FT_VGG, self).__init__()
        self.features = model.features
        self.classifier = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def fc_vgg16( num_classes: int = 20, pretrained: bool = True):
    """FC ft_vgg16.
    """
    model = FT_VGG(torchvision.models.vgg16(pretrained), num_classes)
    return model  

class dilated_conv5_body(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                         padding=1, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(64, 64, kernel_size=3, stride=1,
                                         padding=1, bias=True),
                               nn.ReLU(inplace=True),
                               nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1,
                                         padding=1, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(128, 128, kernel_size=3, stride=1,
                                         padding=1, bias=True),
                               nn.ReLU(inplace=True),
                               nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1,
                                         padding=1, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                         padding=1, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                         padding=1, bias=True),
                               nn.ReLU(inplace=True),
                               nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1,
                                         padding=1, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                         padding=1, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                         padding=1, bias=True),
                               nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                         padding=2, dilation=2, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                         padding=2, dilation=2, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                         padding=2, dilation=2, bias=True),
                               nn.ReLU(inplace=True))

        self.dim_out = 512
        self.spatial_scale = 1. / 8.

        self._init_modules()

    def _init_modules(self):
        assert cfg.VGG.FREEZE_AT in [0, 2, 3, 4, 5]     # 默认不能只冻结conv1的参数 # TODO：why？
        for i in range(1, cfg.VGG.FREEZE_AT + 1):
            freeze_params(getattr(self, 'conv%d' % i))

    def detectron_weight_mapping(self):
        mapping_to_detectron = {
            'conv1.0.weight': 'conv1_0_w',
            'conv1.0.bias': 'conv1_0_b',
            'conv1.2.weight': 'conv1_2_w',
            'conv1.2.bias': 'conv1_2_b',
            'conv2.0.weight': 'conv2_0_w',
            'conv2.0.bias': 'conv2_0_b',
            'conv2.2.weight': 'conv2_2_w',
            'conv2.2.bias': 'conv2_2_b',
            'conv3.0.weight': 'conv3_0_w',
            'conv3.0.bias': 'conv3_0_b',
            'conv3.2.weight': 'conv3_2_w',
            'conv3.2.bias': 'conv3_2_b',
            'conv3.4.weight': 'conv3_4_w',
            'conv3.4.bias': 'conv3_4_b',
            'conv4.0.weight': 'conv4_0_w',
            'conv4.0.bias': 'conv4_0_b',
            'conv4.2.weight': 'conv4_2_w',
            'conv4.2.bias': 'conv4_2_b',
            'conv4.4.weight': 'conv4_4_w',
            'conv4.4.bias': 'conv4_4_b',
            'conv5.0.weight': 'conv5_0_w',
            'conv5.0.bias': 'conv5_0_b',
            'conv5.2.weight': 'conv5_2_w',
            'conv5.2.bias': 'conv5_2_b',
            'conv5.4.weight': 'conv5_4_w',
            'conv5.4.bias': 'conv5_4_b',
        }
        orphan_in_detectron = []

        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        self.training = mode

        for i in range(cfg.VGG.FREEZE_AT + 1, 6):
            getattr(self, 'conv%d' % i).train(mode)

    def forward(self, x):
        for i in range(1, 6):
            x = getattr(self, 'conv%d' % i)(x)
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
        mask_x = box_x * masks.expand(-1,512,-1,-1)

        box_mask_cat = torch.concat((box_x,mask_x),dim=1)
        box_mask_cat = self.mask_branch(box_mask_cat)
        seg_x = self.seg_fc(box_mask_cat.view(batch_size, -1))

        return seg_x

def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False

