import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from core.config import cfg
import nn as mynn
import utils.net as net_utils
from utils.resnet_weights_helper import convert_state_dict
from torchvision import models

# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNet50, ResNet101, ...)
# ---------------------------------------------------------------------------- #

def add_ResNet50_conv4_body():
    return ResNet_convX_body((3, 4, 6))


def add_ResNet50_conv5_body():
  return ResNet_convX_body((3, 4, 6, 3),cfg.ResNet.RES5_DILATION)

def torch_resnet50():
    return resnet()

# ---------------------------------------------------------------------------- #
# Generic ResNet components
# ---------------------------------------------------------------------------- #

class resnet(nn.Module):
    def __init__(self, block_counts=4):
        super(resnet, self).__init__()
        backbone = models.resnet50(pretrained=True)

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

class ResNet_convX_body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts
        self.convX = len(block_counts) + 1
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2

        self.res1 = globals()[cfg.ResNet.STEM_FUNC]()
        dim_in = 64
        dim_bottleneck = cfg.ResNet.NUM_GROUPS * cfg.ResNet.WIDTH_PER_GROUP
        self.res2, dim_in = add_stage(dim_in, 256, dim_bottleneck, block_counts[0],
                                      dilation=1, stride_init=1)
        self.res3, dim_in = add_stage(dim_in, 512, dim_bottleneck * 2, block_counts[1],
                                      dilation=1, stride_init=2)
        # self.res4, dim_in = add_stage(dim_in, 1024, dim_bottleneck * 4, block_counts[2],
        #                               dilation=1, stride_init=2)
        self.res4, dim_in = add_stage(dim_in, 1024, dim_bottleneck * 4, block_counts[2],
                                      dilation=1, stride_init=1)
        if len(block_counts) == 4:
            raise AssertionError
            stride_init = 2 if cfg.ResNet.RES5_DILATION == 1 else 1
            self.res5, dim_in = add_stage(dim_in, 2048, dim_bottleneck * 8, block_counts[3],
                                          cfg.ResNet.RES5_DILATION, stride_init)
            self.spatial_scale = 1 / 32 * cfg.ResNet.RES5_DILATION
        else:
            self.spatial_scale = 1 / 8  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        self._init_modules()

    def _init_modules(self):
        assert cfg.ResNet.FREEZE_AT in [0, 2, 3, 4, 5]
        assert cfg.ResNet.FREEZE_AT <= self.convX
        for i in range(1, cfg.ResNet.FREEZE_AT + 1):
            freeze_params(getattr(self, 'res%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.ResNet.USE_GN:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.gn1.weight': 'conv1_gn_s',
                'res1.gn1.bias': 'conv1_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.bn1.weight': 'res_conv1_bn_s',
                'res1.bn1.bias': 'res_conv1_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'fc1000_w', 'fc1000_b']

        for res_id in range(2, self.convX + 1):
            stage_name = 'res%d' % res_id
            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name,
                self.block_counts[res_id - 2], res_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.ResNet.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'res%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
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

        # self.seg_fc = nn.Sequential(nn.Linear(dim_in * roi_size**2, hidden_dim),
        #                             nn.ReLU(),
        #                             nn.Dropout(),
        #                             nn.Linear(hidden_dim, hidden_dim),
        #                             nn.ReLU(),
        #                             nn.Dropout()
        #                             )

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

        return seg_x, None,None

def add_stage(inplanes, outplanes, innerplanes, nblocks, dilation=1, stride_init=2):
    """Make a stage consist of `nblocks` residual blocks.
    Returns:
        - stage module: an nn.Sequentail module of residual blocks
        - final output dimension
    """
    res_blocks = []
    stride = stride_init
    for _ in range(nblocks):
        res_blocks.append(add_residual_block(
            inplanes, outplanes, innerplanes, dilation, stride
        ))
        inplanes = outplanes
        stride = 1

    return nn.Sequential(*res_blocks), outplanes


def add_residual_block(inplanes, outplanes, innerplanes, dilation, stride):
    """Return a residual block module, including residual connection, """
    if stride != 1 or inplanes != outplanes:
        shortcut_func = globals()[cfg.ResNet.SHORTCUT_FUNC]
        downsample = shortcut_func(inplanes, outplanes, stride)
    else:
        downsample = None

    trans_func = globals()[cfg.ResNet.TRANS_FUNC]
    res_block = trans_func(
        inplanes, outplanes, innerplanes, stride,
        dilation=dilation, group=cfg.ResNet.NUM_GROUPS,
        downsample=downsample)

    return res_block


# ------------------------------------------------------------------------------
# various downsample shortcuts (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes,
                  outplanes,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        mynn.AffineChannel2d(outplanes),
    )


def basic_gn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes,
                  outplanes,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes,
                     eps=cfg.GROUP_NORM.EPSILON)
    )


# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

# this function has been edited
def basic_bn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
        ('bn1', mynn.AffineChannel2d(64)),
        ('relu', nn.ReLU(inplace=True)),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))


def basic_gn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
        ('gn1', nn.GroupNorm(net_utils.get_group_gn(64), 64,
                             eps=cfg.GROUP_NORM.EPSILON)),
        ('relu', nn.ReLU(inplace=True)),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

class bottleneck_transformation(nn.Module):
    """ Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.ResNet.STRIDE_1X1 else (1, stride)
        self.stride = stride

        self.conv1 = nn.Conv2d(
            inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.bn1 = mynn.AffineChannel2d(innerplanes)

        self.conv2 = nn.Conv2d(
            innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
            padding=1 * dilation, dilation=dilation, groups=group)
        self.bn2 = mynn.AffineChannel2d(innerplanes)

        self.conv3 = nn.Conv2d(
            innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.bn3 = mynn.AffineChannel2d(outplanes)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class bottleneck_gn_transformation(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.ResNet.STRIDE_1X1 else (1, stride)
        self.stride = stride

        self.conv1 = nn.Conv2d(
            inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.gn1 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.conv2 = nn.Conv2d(
            innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
            padding=1 * dilation, dilation=dilation, groups=group)
        self.gn2 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.conv3 = nn.Conv2d(
            innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.gn3 = nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #

def residual_stage_detectron_mapping(module_ref, module_name, num_blocks, res_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `res_id`
    """
    if cfg.ResNet.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for blk_id in range(num_blocks):
        detectron_prefix = 'res%d_%d' % (res_id, blk_id)
        my_prefix = '%s.%d' % (module_name, blk_id)

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = detectron_prefix + '_branch1'  # short for "detectron_branch_prefix"
            mapping_to_detectron[my_prefix
                                 + '.downsample.0.weight'] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.weight'] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.bias'] = dtt_bp + norm_suffix + '_b'

        # conv branch
        for i, c in zip([1, 2, 3], ['a', 'b', 'c']):
            dtt_bp = detectron_prefix + '_branch2' + c
            mapping_to_detectron[my_prefix
                                 + '.conv%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix
                                 + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix
                                 + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'

    return mapping_to_detectron, orphan_in_detectron


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
