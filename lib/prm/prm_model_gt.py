# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from typing import Union, Optional, List, Tuple
from tqdm import tqdm

# %%
from types import MethodType
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.prm.prm_modules import pr_conv2d, peak_stimulation, peak_stimulation_aff
from mmcv.ops import RoIAlign
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
import os
from PIL import Image
from pycocotools import mask as maskUtils

VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')


def normalize_cam(class_response_maps_normalize):
    class_response_maps_normalize = (class_response_maps_normalize - class_response_maps_normalize.min((-2, -1))[
        ..., None, None]) / \
                                    (1e-5 + class_response_maps_normalize.max((-2, -1))[..., None, None] -
                                     class_response_maps_normalize.min((-2, -1))[..., None, None])

    print(class_response_maps_normalize)
    return class_response_maps_normalize


def get_sp_f(box_feature, mask, resolution):
    sp_f = (mask * box_feature).view(-1, resolution, resolution)  # (1,dim,7,7) --> (dim,7,7)
    sp_f = sp_f.sum((-2, -1))  # (dim)
    sp_f = sp_f / (mask.sum() + 1e-5)

    return sp_f[None, :]


class PeakResponseMapping(nn.Sequential):

    def __init__(self, *args, **kargs):
        super(PeakResponseMapping, self).__init__(*args)

        self.inferencing = False
        # use global average pooling to aggregate responses if peak stimulation is disabled
        self.enable_peak_stimulation = kargs.get('enable_peak_stimulation', True)
        # return only the class response maps in inference mode if peak backpropagation is disabled
        self.enable_peak_backprop = kargs.get('enable_peak_backprop', True)
        # window size for peak finding
        self.win_size = kargs.get('win_size', 3)
        # sub-pixel peak finding
        self.sub_pixel_locating_factor = kargs.get('sub_pixel_locating_factor', 1)
        # peak filtering
        self.filter_type = kargs.get('filter_type', 'median')
        if self.filter_type == 'median':
            self.peak_filter = self._median_filter
        elif self.filter_type == 'mean':
            self.peak_filter = self._mean_filter
        elif self.filter_type == 'max':
            self.peak_filter = self._max_filter
        elif isinstance(self.filter_type, (int, float)):
            self.peak_filter = lambda x: self.filter_type
        else:
            self.peak_filter = None

    @staticmethod
    def _median_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _mean_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _max_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward

    def instance_nms(self, instance_list, threshold=0.3, merge_peak_response=True):
        selected_instances = []
        while len(instance_list) > 0:
            instance = instance_list.pop(0)
            selected_instances.append(instance)
            src_mask = instance[2].astype(bool)
            src_peak_response = instance[3]

            def iou_filter(x):
                dst_mask = x[2].astype(bool)
                # IoU
                intersection = np.logical_and(src_mask, dst_mask).sum()
                union = np.logical_or(src_mask, dst_mask).sum()
                iou = intersection / (union + 1e-10)
                if iou < threshold:
                    return x
                else:
                    if merge_peak_response:
                        nonlocal src_peak_response
                        src_peak_response += x[3]
                    return None

            instance_list = list(filter(iou_filter, instance_list))
        return selected_instances

    def instance_seg(self, class_response_maps, peak_list, peak_response_maps, retrieval_cfg):
        # cast tensors to numpy array
        class_response_maps = class_response_maps.squeeze().cpu().numpy()
        peak_list = peak_list.cpu().numpy()
        peak_response_maps = peak_response_maps.cpu().numpy()

        img_height, img_width = peak_response_maps.shape[1], peak_response_maps.shape[2]

        # image size
        img_area = img_height * img_width

        # segment proposals off-the-shelf
        proposals = retrieval_cfg['proposals']

        # proposal contour width
        contour_width = retrieval_cfg.get('contour_width', 5)

        # limit range of proposal size
        proposal_size_limit = retrieval_cfg.get('proposal_size_limit', (0.00002, 0.85))

        # selected number of proposals
        proposal_count = retrieval_cfg.get('proposal_count', 100)

        # nms threshold
        nms_threshold = retrieval_cfg.get('nms_threshold', 0.3)

        # merge peak response during nms
        merge_peak_response = retrieval_cfg.get('merge_peak_response', True)

        # metric free parameters
        param = retrieval_cfg.get('param', None)

        # process each peak
        instance_list = []
        for i in range(len(peak_response_maps)):
            class_idx = peak_list[i, 1]

            # extract hyper-params
            if isinstance(param, tuple):
                # shared param
                bg_threshold_factor, penalty_factor, balance_factor = param
            elif isinstance(param, list):
                # independent params between classes
                bg_threshold_factor, penalty_factor, balance_factor = param[class_idx]
            else:
                raise TypeError('Invalid hyper-params "%s".' % param)

            class_response = imresize(class_response_maps[class_idx], (img_height, img_width), interp='bicubic')
            bg_response = (class_response < bg_threshold_factor * class_response.mean()).astype(np.float32)
            peak_response_map = peak_response_maps[i]

            # select proposal
            max_val = -np.inf
            instance_mask = None

            for j in range(min(proposal_count, len(proposals))):
                raw_mask = imresize(proposals[j].astype(int), peak_response_map.shape, interp='nearest')
                # get contour of the proposal
                contour_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_GRADIENT,
                                                np.ones((contour_width, contour_width), np.uint8)).astype(bool)
                mask = raw_mask.astype(bool)
                # metric
                mask_area = mask.sum()
                if (mask_area >= proposal_size_limit[1] * img_area) or \
                        (mask_area < proposal_size_limit[0] * img_area):
                    continue
                else:
                    val = balance_factor * peak_response_map[mask].sum() + \
                          peak_response_map[contour_mask].sum() - \
                          penalty_factor * bg_response[mask].sum()
                    if val > max_val:
                        max_val = val
                        instance_mask = mask

            if instance_mask is not None:
                instance_list.append((max_val, class_idx, instance_mask, peak_response_map))

        instance_list = sorted(instance_list, key=lambda x: x[0], reverse=True)
        if nms_threshold is not None:
            instance_list = self.instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), nms_threshold,
                                              merge_peak_response)
        return [dict(score=v[0], category=v[1], mask=v[2], prm=v[3]) for v in instance_list]

    def forward(self, input, lables, class_threshold=0, peak_threshold=10, retrieval_cfg=None):
        assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
        if self.inferencing:
            input.requires_grad_()

        # classification network forwarding

        class_response_maps = super(PeakResponseMapping, self).forward(input)

        if self.enable_peak_stimulation:
            # sub-pixel peak finding
            if self.sub_pixel_locating_factor > 1:
                class_response_maps = F.upsample(class_response_maps, scale_factor=self.sub_pixel_locating_factor,
                                                 mode='bilinear', align_corners=True)
            # aggregate responses from informative receptive fields estimated via class peak responses
            peak_list, aggregation = peak_stimulation(class_response_maps, win_size=self.win_size,
                                                      peak_filter=self.peak_filter)
        else:
            # aggregate responses from all receptive fields
            peak_list, aggregation = None, F.adaptive_avg_pool2d(class_response_maps, 1).squeeze(2).squeeze(2)

        if self.inferencing:
            if not self.enable_peak_backprop:
                # extract only class-aware visual cues
                return aggregation, class_response_maps

            # extract instance-aware visual cues, i.e., peak response maps
            assert class_response_maps.size(
                0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'
            if peak_list is None:
                peak_list = peak_stimulation(class_response_maps, return_aggregation=False, win_size=self.win_size,
                                             peak_filter=self.peak_filter)

            peak_response_maps = []
            valid_peak_list = []
            peak_score = []
            # peak backpropagation
            grad_output = class_response_maps.new_empty(class_response_maps.size())

            # print(peak_list)
            # for each gt label
            for cls_idx in lables:
                peak_list_cls = peak_list[peak_list[:, 1] == cls_idx]
                # print(peak_list_cls)
                exist = 0
                for idx in range(peak_list_cls.size(0)):
                    peak_val = class_response_maps[
                        peak_list_cls[idx, 0], peak_list_cls[idx, 1], peak_list_cls[idx, 2], peak_list_cls[idx, 3]]
                    # print(peak_val)
                    if peak_val > peak_threshold:
                        exist = 1
                        grad_output.zero_()
                        # starting from the peak
                        grad_output[peak_list_cls[idx, 0], peak_list_cls[idx, 1], peak_list_cls[idx, 2], peak_list_cls[
                            idx, 3]] = 1
                        if input.grad is not None:
                            input.grad.zero_()
                        class_response_maps.backward(grad_output, retain_graph=True)
                        prm = input.grad.detach().sum(1).clone().clamp(min=0)
                        peak_response_maps.append(prm / prm.sum())
                        valid_peak_list.append(peak_list_cls[idx, :])
                        peak_score.append(peak_val)
                if exist == 0:
                    peak_val_list = [class_response_maps[0, cls_idx, peak_list_cls[idx, 2], peak_list_cls[idx, 3]] for
                                     idx in range(peak_list_cls.size(0))]
                    peak_val_list = torch.tensor(peak_val_list)
                    peak_val = class_response_maps[peak_list_cls[peak_val_list.argmax(), 0], \
                                                   peak_list_cls[peak_val_list.argmax(), 1], \
                                                   peak_list_cls[peak_val_list.argmax(), 2], \
                                                   peak_list_cls[peak_val_list.argmax(), 3]]
                    grad_output.zero_()
                    # starting from the peak
                    grad_output[peak_list_cls[peak_val_list.argmax(), 0], \
                                peak_list_cls[peak_val_list.argmax(), 1], \
                                peak_list_cls[peak_val_list.argmax(), 2], \
                                peak_list_cls[peak_val_list.argmax(), 3]] = 1
                    if input.grad is not None:
                        input.grad.zero_()
                    class_response_maps.backward(grad_output, retain_graph=True)
                    prm = input.grad.detach().sum(1).clone().clamp(min=0)
                    peak_response_maps.append(prm / prm.sum())
                    valid_peak_list.append(peak_list_cls[peak_val_list.argmax(), :])
                    peak_score.append(peak_val)

            # return results
            class_response_maps = class_response_maps.detach()
            aggregation = aggregation.detach()

            if len(peak_response_maps) > 0:
                valid_peak_list = torch.stack(valid_peak_list)
                peak_response_maps = torch.cat(peak_response_maps, 0)
                peak_score = torch.tensor(peak_score)
                if retrieval_cfg is None:
                    # classification confidence scores, class-aware and instance-aware visual cues
                    return aggregation, class_response_maps, valid_peak_list, peak_response_maps, peak_score
                else:
                    # instance segmentation using build-in proposal retriever
                    return self.instance_seg(class_response_maps, valid_peak_list, peak_response_maps, retrieval_cfg)
            else:
                return None  # aggregation, class_response_maps # None
        else:
            # classification confidence scores
            return aggregation, peak_list, class_response_maps

    def train(self, mode=True):
        super(PeakResponseMapping, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super(PeakResponseMapping, self).train(False)
        self._patch()
        self.inferencing = True
        return self


# %%
class FC_ResNet(nn.Module):

    def __init__(self, model, num_classes):
        super(FC_ResNet, self).__init__()

        # feature encoding
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classifier
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def fc_resnet50(num_classes: int = 20, pretrained: bool = True) -> nn.Module:
    """FC ResNet50.
    """
    model = FC_ResNet(models.resnet50(pretrained), num_classes)
    return model


def peak_response_mapping(
        backbone: nn.Module = fc_resnet50(),
        enable_peak_stimulation: bool = True,
        enable_peak_backprop: bool = True,
        win_size: int = 3,
        sub_pixel_locating_factor: int = 8,
        filter_type: Union[str, int, float] = 'median') -> nn.Module:
    """Peak Response Mapping.
    """

    model = PeakResponseMapping(
        backbone,
        enable_peak_stimulation=enable_peak_stimulation,
        enable_peak_backprop=enable_peak_backprop,
        win_size=win_size,
        sub_pixel_locating_factor=sub_pixel_locating_factor,
        filter_type=filter_type)
    return model


voc_id_name_map = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
                   9: 'chair', 10: 'cow',
                   11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant',
                   17: 'sheep', 18: 'sofa',
                   19: 'train', 20: 'tvmonitor'}


def imresize(arr, size, interp='bilibear', mode=None):
    im = Image.fromarray(np.uint8(arr), mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size) * percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'biliear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])  # 调用PIL库中的resize函数
    return np.array(imnew)


# copy from IAM
# get low level feature
class FC_Resnet_aff(nn.Module):
    def __init__(self, model, num_classes):
        super(FC_Resnet_aff, self).__init__()

        # feature encoding
        self.res_block1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool)

        self.res_block2 = model.layer1
        self.res_block3 = model.layer2
        self.res_block4 = model.layer3
        self.res_block5 = model.layer4

        # classifier
        self.num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(self.num_features, num_classes, kernel_size=1, bias=True))

    def forward(self, x):
        p1 = self.res_block1(x)
        p2 = self.res_block2(p1)
        p3 = self.res_block3(p2)
        p4 = self.res_block4(p3)
        p5 = self.res_block5(p4)

        class_response_maps = self.classifier(p5)

        return class_response_maps, p1, p2, p3, p4


def fc_resnet50_aff(num_classes: int = 20, pretrained: bool = True) -> nn.Module:
    model = FC_Resnet_aff(models.resnet50(pretrained), num_classes)
    return model


def peak_response_mapping_aff(
        backbone: nn.Module = fc_resnet50_aff(),
        enable_peak_stimulation: bool = True,
        enable_peak_backprop: bool = True,
        win_size: int = 3,
        sub_pixel_locating_factor: int = 8,
        filter_type: Union[str, int, float] = 'median') -> nn.Module:
    """Peak Response Mapping.
    """

    model = PeakResponseMapping_aff(
        backbone,
        enable_peak_stimulation=enable_peak_stimulation,
        enable_peak_backprop=enable_peak_backprop,
        win_size=win_size,
        sub_pixel_locating_factor=sub_pixel_locating_factor,
        filter_type=filter_type)
    return model


# only for inferencing
class PeakResponseMapping_aff(nn.Sequential):

    def __init__(self, *args, **kargs):
        super(PeakResponseMapping_aff, self).__init__(*args)

        self.inferencing = True
        # use global average pooling to aggregate responses if peak stimulation is disabled
        self.enable_peak_stimulation = kargs.get('enable_peak_stimulation', True)
        # return only the class response maps in inference mode if peak backpropagation is disabled
        self.enable_peak_backprop = kargs.get('enable_peak_backprop', True)
        # window size for peak finding
        self.win_size = kargs.get('win_size', 3)
        # sub-pixel peak finding
        self.sub_pixel_locating_factor = kargs.get('sub_pixel_locating_factor', 1)
        # peak filtering
        self.filter_type = kargs.get('filter_type', 'median')
        if self.filter_type == 'median':
            self.peak_filter = self._median_filter
        elif self.filter_type == 'mean':
            self.peak_filter = self._mean_filter
        elif self.filter_type == 'max':
            self.peak_filter = self._max_filter
        elif isinstance(self.filter_type, (int, float)):
            self.peak_filter = lambda x: self.filter_type
        else:
            self.peak_filter = None

    @staticmethod
    def _median_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _mean_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _max_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _lzc_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return 0.5 * threshold.contiguous().view(batch_size, num_channels, 1, 1)

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward

    def instance_nms(self, instance_list, threshold=0.3, merge_peak_response=True):
        selected_instances = []
        while len(instance_list) > 0:
            instance = instance_list.pop(0)
            selected_instances.append(instance)
            src_mask = instance[2].astype(bool)
            src_peak_response = instance[3]

            def iou_filter(x):
                dst_mask = x[2].astype(bool)
                # IoU
                intersection = np.logical_and(src_mask, dst_mask).sum()
                union = np.logical_or(src_mask, dst_mask).sum()
                iou = intersection / (union + 1e-10)
                if iou < threshold:
                    return x
                else:
                    if merge_peak_response:
                        nonlocal src_peak_response
                        src_peak_response += x[3]
                    return None

            instance_list = list(filter(iou_filter, instance_list))
        return selected_instances

    def instance_seg(self, class_response_maps, peak_list, peak_response_maps, retrieval_cfg):
        # cast tensors to numpy array
        class_response_maps = class_response_maps.squeeze().cpu().numpy()
        peak_list = peak_list.cpu().numpy()
        peak_response_maps = peak_response_maps.cpu().numpy()

        img_height, img_width = peak_response_maps.shape[1], peak_response_maps.shape[2]

        # image size
        img_area = img_height * img_width

        # segment proposals off-the-shelf
        proposals = retrieval_cfg['proposals']

        # proposal contour width
        contour_width = retrieval_cfg.get('contour_width', 5)

        # limit range of proposal size
        proposal_size_limit = retrieval_cfg.get('proposal_size_limit', (0.00002, 0.85))

        # selected number of proposals
        proposal_count = retrieval_cfg.get('proposal_count', 100)

        # nms threshold
        nms_threshold = retrieval_cfg.get('nms_threshold', 0.3)

        # merge peak response during nms
        merge_peak_response = retrieval_cfg.get('merge_peak_response', True)

        # metric free parameters
        param = retrieval_cfg.get('param', None)

        # process each peak
        instance_list = []
        for i in range(len(peak_response_maps)):
            class_idx = peak_list[i, 1]

            # extract hyper-params
            if isinstance(param, tuple):
                # shared param
                bg_threshold_factor, penalty_factor, balance_factor = param
            elif isinstance(param, list):
                # independent params between classes
                bg_threshold_factor, penalty_factor, balance_factor = param[class_idx]
            else:
                raise TypeError('Invalid hyper-params "%s".' % param)

            class_response = imresize(class_response_maps[class_idx], (img_height, img_width), interp='bicubic')
            bg_response = (class_response < bg_threshold_factor * class_response.mean()).astype(np.float32)
            peak_response_map = peak_response_maps[i]

            # select proposal
            max_val = -np.inf
            instance_mask = None

            for j in range(min(proposal_count, len(proposals))):
                raw_mask = imresize(proposals[j].astype(int), peak_response_map.shape, interp='nearest')
                # get contour of the proposal
                contour_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_GRADIENT,
                                                np.ones((contour_width, contour_width), np.uint8)).astype(bool)
                mask = raw_mask.astype(bool)
                # metric
                mask_area = mask.sum()
                if (mask_area >= proposal_size_limit[1] * img_area) or \
                        (mask_area < proposal_size_limit[0] * img_area):
                    continue
                else:
                    val = balance_factor * peak_response_map[mask].sum() + \
                          peak_response_map[contour_mask].sum() - \
                          penalty_factor * bg_response[mask].sum()
                    if val > max_val:
                        max_val = val
                        instance_mask = mask

            if instance_mask is not None:
                instance_list.append((max_val, class_idx, instance_mask, peak_response_map))

        instance_list = sorted(instance_list, key=lambda x: x[0], reverse=True)
        if nms_threshold is not None:
            instance_list = self.instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), nms_threshold,
                                              merge_peak_response)
        return [dict(score=v[0], category=v[1], mask=v[2], prm=v[3]) for v in instance_list]

    # # cam normalize to RW
    # def forward(self, input, lables, img,class_threshold=0, peak_threshold=10, retrieval_cfg=None,graph_op=None,resolution=7):
    #     assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
    #     dir = "./voc_visual"
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #     with torch.no_grad():
    #
    #         # _, _, H, W = input.shape  # batchsize == 1
    #         H,W,_ = np.array(img).shape
    #
    #         LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #         HSV_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #
    #         # classification network forwarding
    #         class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
    #
    #         _, cls_num, cam_H, cam_W = class_response_maps.shape
    #
    #         onehot_label = torch.tensor(self.label2onehot(lables,cls_num))
    #         class_response_maps = class_response_maps * torch.tensor(onehot_label).view(1,cls_num,1,1)
    #
    #         class_response_maps = class_response_maps.detach()
    #         class_response_maps_normalize = class_response_maps.numpy()
    #         class_response_maps_normalize = normalize_cam(class_response_maps_normalize)
    #
    #         class_response_maps_normalize = torch.tensor(class_response_maps_normalize)
    #         org_size_class_response_maps = F.upsample(class_response_maps_normalize, size=(H,W), mode='bilinear', align_corners=True)
    #
    #         # use p2 as feature
    #         feature = p4
    #
    #         _, _, f_H, f_W = feature.shape
    #
    #         # spatial_scale
    #         spatial_scale = f_H / H
    #         Align_op = RoIAlign((resolution, resolution), spatial_scale, 0)
    #
    #         # graph operate
    #         sp_label = graph_op["sp_label"]
    #         adjacent_matrix = graph_op["adjacent_matrix"]
    #         img_id = graph_op["img_id"]
    #         belong_mat = graph_op["belong_mat"]
    #         proposal_score = graph_op["proposal_score"]
    #         print(img_id)
    #
    #         # add bg_score
    #         bg_score = 1 - proposal_score.sum(-1)
    #         proposal_score = np.concatenate((bg_score[:,None],proposal_score),axis=1)
    #
    #         sp_scores = np.zeros((adjacent_matrix.shape[0],proposal_score.shape[-1])) # sp_num, cls_num + 1
    #         for row_idx in range(belong_mat.shape[0]):
    #             score = proposal_score[row_idx]
    #             belong_row = belong_mat[row_idx] != 0
    #
    #             sp_scores[belong_row] += score
    #
    #         aff_proposal_score = torch.tensor(sp_scores)
    #         low_label = np.min(sp_label)
    #         high_label = np.max(sp_label)
    #
    #         for idx, temp_label in tqdm(enumerate(range(low_label, high_label + 1)),total=adjacent_matrix.shape[0]):
    #             sp_mask = sp_label == temp_label
    #
    #             sp_mask_tensor = torch.tensor(sp_mask)
    #             cam_scores = (sp_mask_tensor * org_size_class_response_maps).sum((-2,-1)) / sp_mask_tensor.sum() # --> (1, cls_num)
    #
    #             ind_xy = np.nonzero(sp_mask)
    #             xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
    #
    #             mask = sp_mask[ymin:ymax, xmin:xmax]
    #             mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
    #             mask = torch.tensor(mask)
    #
    #             box = torch.tensor([[0, xmin, ymin, xmax, ymax]],dtype=torch.float32)
    #             box_feature = Align_op(feature, box)  # (1,7,7)
    #
    #             sp_f = (mask * box_feature).view(-1, resolution, resolution)  # (1,dim,7,7) --> (dim,7,7)
    #             sp_f = sp_f.sum((-2,-1))  # (dim)
    #             sp_f = sp_f / (mask.sum()+1e-5)
    #
    #             sp_f = sp_f[None, :]
    #
    #             if idx == 0:
    #                 sp_cams = cam_scores
    #                 sp_features = sp_f
    #
    #                 color_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                 for color_idx in range(1,3):
    #                     temp_feature = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     color_feature = np.concatenate((color_feature,temp_feature),axis=0)
    #
    #                 rgb_features = color_feature.transpose()
    #
    #             else:
    #                 sp_cams = torch.cat((sp_cams, cam_scores), dim=0)
    #                 sp_features = torch.cat((sp_features, sp_f), dim=0)
    #
    #                 color_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                 for color_idx in range(1,3):
    #                     temp_feature = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     color_feature = np.concatenate((color_feature,temp_feature),axis=0)
    #
    #                 rgb_features = np.concatenate((rgb_features,color_feature.transpose()),axis=0)
    #
    #         rgb_features = torch.tensor(rgb_features)
    #
    #         # make similarity
    #         aff_sp_f = torch.nn.functional.cosine_similarity(sp_features.unsqueeze(0), sp_features.unsqueeze(1), dim=-1)
    #         aff_score = torch.nn.functional.cosine_similarity(aff_proposal_score.unsqueeze(0), aff_proposal_score.unsqueeze(1), dim=-1)
    #         aff_rgb = torch.nn.functional.cosine_similarity(rgb_features.unsqueeze(0), rgb_features.unsqueeze(1), dim=-1)
    #         # belong_mat = torch.tensor(belong_mat).t()
    #         # aff_belong = torch.nn.functional.cosine_similarity(belong_mat.unsqueeze(0),belong_mat.unsqueeze(1),dim=-1)
    #
    #         aff_mat = aff_sp_f + aff_score + aff_rgb
    #
    #         aff_mat_mask = aff_mat * (torch.tensor(adjacent_matrix, dtype=torch.float32) + torch.eye(adjacent_matrix.shape[0]))
    #
    #         trans_mat = aff_mat_mask / (torch.sum(aff_mat_mask, dim=0, keepdim=True)+1e-5)
    #
    #         # logt = 4
    #         #
    #         # for _ in range(logt):
    #         #     trans_mat = torch.matmul(trans_mat, trans_mat)
    #
    #         # trans_mat = torch.linalg.inv(1e-5 * torch.eye(aff_mat.shape[0]) + torch.tensor(aff_mat_mask,dtype=torch.float32)) @ torch.tensor(aff_mat_mask,dtype=torch.float32)
    #
    #         plt.close()
    #         fig,axx = plt.subplots(2, 3)
    #
    #         sns.heatmap(aff_sp_f.numpy(),cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,0])
    #         axx[0,0].set_title("aff_sp_f")
    #
    #         sns.heatmap(aff_score.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,1])
    #         axx[0,1].set_title("aff_score")
    #
    #         # axx[0,2].axis("off")
    #         sns.heatmap(aff_rgb.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,2])
    #         axx[0,2].set_title("aff_rgb")
    #
    #         sns.heatmap(aff_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,0])
    #         axx[1,0].set_title("aff_mat")
    #
    #         sns.heatmap(aff_mat_mask.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,1])
    #         axx[1,1].set_title("aff_mat_mask")
    #
    #         sns.heatmap(trans_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,2])
    #         axx[1,2].set_title("trans_mat")
    #
    #         print(os.path.join(dir,str(img_id)+"_aff.jpg"))
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+"_aff.jpg"), dpi=300)
    #
    #         sp_cams = sp_cams.numpy() - sp_cams.numpy().min(0)
    #         sp_cams = sp_cams / (1e-5 + sp_cams.max(0))
    #         sp_cams = torch.tensor(sp_cams)
    #         # print(sp_cams)
    #
    #         RW_cams = torch.tensor(sp_cams.t(), dtype=torch.float32)
    #         trans_mat = torch.tensor(trans_mat, dtype=torch.float32)
    #
    #         cam_valid_peak_list, cam_peak_score = self.get_peak_list(class_response_maps,lables,peak_threshold)
    #         response_maps = [class_response_maps]
    #         valid_peak_list = [cam_valid_peak_list]
    #         scale = 0.7
    #
    #         iter_TIME = 4
    #         alph = 0.7
    #
    #         for _ in range(iter_TIME):
    #             RW_cams = alph * torch.matmul(RW_cams, trans_mat) + (1 - alph) * torch.tensor(sp_cams.t(),dtype=torch.float32)
    #             RW_cams = RW_cams.numpy() - RW_cams.numpy().min(0)
    #             RW_cams = RW_cams / (1e-5 + RW_cams.max(0))
    #             RW_cams = torch.tensor(RW_cams)
    #
    #             RW_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_cams.dtype)
    #
    #             for cls_idx in lables:
    #                 for idx, temp_label in enumerate(range(low_label, high_label + 1)):
    #                     sp_mask = sp_label == temp_label
    #                     sp_mask_tensor = torch.tensor(sp_mask)
    #                     RW_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_cams[cls_idx,temp_label] # single batch input
    #
    #             RW_class_response_maps = F.upsample(RW_class_response_maps,size=(cam_H,cam_W),mode='bilinear', align_corners=True)
    #
    #             RW_valid_peak_list, RW_peak_score = self.get_peak_list(RW_class_response_maps,lables,0.9)
    #
    #             response_maps.append(RW_class_response_maps)
    #             valid_peak_list.append(RW_valid_peak_list)
    #
    #         cam_num = onehot_label.sum().item()
    #         plt.close()
    #         f, axarr = plt.subplots(len(response_maps), int(cam_num + 1))
    #         plt.tight_layout()
    #
    #         for idx in range(len(response_maps)):
    #             if idx == 0:
    #                 axarr[idx, 0].imshow(img)
    #             else:
    #                 axarr[idx, 0].imshow(mark_boundaries(img, sp_label))
    #
    #             axarr[idx, 0].axis("off")
    #
    #             for i,cls_idx in enumerate(lables):
    #                 CRM = response_maps[idx]
    #
    #                 cam = CRM[0, cls_idx].detach().cpu().numpy()
    #                 cam = cam - np.min(cam)
    #                 cam = cam / np.max(cam)
    #                 cam = np.uint8(255 * cam)
    #
    #                 heatmap = cv2.applyColorMap(cv2.resize(cam, (W, H)), cv2.COLORMAP_JET)
    #                 heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #                 result = heatmap * scale + (img) * (1 - scale)
    #                 result = np.uint8(result)
    #                 result = cv2.resize(result, (int(cam_W*self.sub_pixel_locating_factor),int(cam_H*self.sub_pixel_locating_factor)))
    #
    #                 axarr[idx, i+1].imshow(result)
    #                 axarr[idx, i+1].axis("off")
    #                 axarr[idx, i+1].set_title(voc_id_name_map[cls_idx.item()+1])
    #
    #                 VPL = valid_peak_list[idx]
    #
    #                 for peak in VPL:
    #                     if peak[1] == cls_idx:
    #                         axarr[idx, i+1].scatter(peak[3],peak[2], color='purple',marker='*',edgecolors='purple', s=17)
    #
    #                 axarr[idx, i+1].axis("off")
    #
    #         # plt.show()
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+".jpg"), dpi=300)
    #
    #         # if len(peak_score) > 0:
    #         #     valid_peak_list = torch.stack(valid_peak_list)
    #         #     peak_score = torch.tensor(peak_score)
    #         #     # classification confidence scores, class-aware and instance-aware visual cues
    #         #     return class_response_maps, valid_peak_list, peak_score
    #         # else:
    #         #     return None

    # # use proposal score to replace cam score
    # # draw
    # def forward(self, input, lables, img,class_threshold=0, peak_threshold=10, retrieval_cfg=None,graph_op=None,resolution=7):
    #     assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
    #     dir = "./voc_visual"
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #     with torch.no_grad():
    #
    #         # _, _, H, W = input.shape  # batchsize == 1
    #         H,W,_ = np.array(img).shape
    #
    #         LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #         HSV_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #
    #         # classification network forwarding
    #         class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
    #         class_response_maps = class_response_maps.detach()
    #         org_size_class_response_maps = F.upsample(class_response_maps, size=(H,W), mode='bilinear', align_corners=True)
    #
    #         _, cls_num, cam_H, cam_W = class_response_maps.shape
    #
    #         onehot_label = self.label2onehot(lables,cls_num)
    #
    #         _, _, f_H_p1, f_W_p1 = p1.shape
    #         _, _, f_H_p2, f_W_p2 = p2.shape
    #         _, _, f_H_p3, f_W_p3 = p3.shape
    #         _, _, f_H_p4, f_W_p4 = p4.shape
    #
    #         Align_op_p1 = RoIAlign((resolution, resolution), f_H_p1/H, 0)
    #         Align_op_p2 = RoIAlign((resolution, resolution), f_H_p2/H, 0)
    #         Align_op_p3 = RoIAlign((resolution, resolution), f_H_p3/H, 0)
    #         Align_op_p4 = RoIAlign((resolution, resolution), f_H_p4/H, 0)
    #
    #         # graph operate
    #         sp_label = graph_op["sp_label"]
    #         adjacent_matrix = graph_op["adjacent_matrix"]
    #         img_id = graph_op["img_id"]
    #         belong_mat = graph_op["belong_mat"]
    #         proposal_score = graph_op["proposal_score"]
    #         print(img_id)
    #
    #         # add bg_score
    #         bg_score = 1 - proposal_score.sum(-1)
    #         proposal_score = np.concatenate((bg_score[:,None],proposal_score),axis=1)
    #         # proposal_score = np.concatenate((bg_score[:,None],proposal_score*onehot_label),axis=1)
    #
    #         # # lzc edit 9-6
    #         # max_idx = np.argmax(proposal_score,axis=-1)
    #         # flag = np.zeros_like(max_idx)
    #         # for label in lables:
    #         #     flag += (label+1).item() == max_idx
    #         #
    #         # bg_score_const = np.zeros(proposal_score.shape[-1])
    #         # bg_score_const[0] = 1
    #         # proposal_score[flag == 0] = bg_score_const
    #         # print(flag)
    #
    #         sp_scores = np.zeros((adjacent_matrix.shape[0],proposal_score.shape[-1])) # sp_num, cls_num + 1
    #         for row_idx in range(belong_mat.shape[0]):
    #             score = proposal_score[row_idx]
    #             belong_row = belong_mat[row_idx] != 0
    #
    #             sp_scores[belong_row] += score
    #
    #         sp_scores = torch.tensor(sp_scores)
    #         sp_scores_normalize = torch.nn.functional.normalize(sp_scores,p=1,dim=-1)
    #
    #         low_label = np.min(sp_label)
    #         high_label = np.max(sp_label)
    #
    #         for idx, temp_label in tqdm(enumerate(range(low_label, high_label + 1)),total=adjacent_matrix.shape[0]):
    #             sp_mask = sp_label == temp_label
    #             if idx == 0:
    #                 sp_proposal = sp_mask[None,:,:]
    #             else:
    #                 sp_proposal = np.concatenate((sp_proposal, sp_mask[None,:,:]),axis=0)
    #
    #             # sp_mask_tensor = torch.tensor(sp_mask)
    #
    #             ind_xy = np.nonzero(sp_mask)
    #             xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
    #
    #             mask = sp_mask[ymin:ymax, xmin:xmax]
    #             mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
    #             mask = torch.tensor(mask)
    #
    #             box = torch.tensor([[0, xmin, ymin, xmax, ymax]],dtype=torch.float32)
    #             box_feature_p1 = Align_op_p1(p1, box)  # (1,7,7)
    #             box_feature_p2 = Align_op_p2(p2, box)  # (1,7,7)
    #             box_feature_p3 = Align_op_p3(p3, box)  # (1,7,7)
    #             box_feature_p4 = Align_op_p4(p4, box)  # (1,7,7)
    #
    #             sp_feature_p1 = get_sp_f(box_feature_p1,mask,resolution)
    #             sp_feature_p2 = get_sp_f(box_feature_p2,mask,resolution)
    #             sp_feature_p3 = get_sp_f(box_feature_p3,mask,resolution)
    #             sp_feature_p4 = get_sp_f(box_feature_p4,mask,resolution)
    #
    #             if idx == 0:
    #                 sp_features_p1 = sp_feature_p1
    #                 sp_features_p2 = sp_feature_p2
    #                 sp_features_p3 = sp_feature_p3
    #                 sp_features_p4 = sp_feature_p4
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1,3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature,temp_feature_rgb),axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature,temp_feature_lab),axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature,temp_feature_hsv),axis=0)
    #
    #                 RGB_features = RGB_feature.transpose()
    #                 LAB_features = LAB_feature.transpose()
    #                 HSV_features = HSV_feature.transpose()
    #
    #             else:
    #                 sp_features_p1 = torch.cat((sp_features_p1, sp_feature_p1),dim=0)
    #                 sp_features_p2 = torch.cat((sp_features_p2, sp_feature_p2),dim=0)
    #                 sp_features_p3 = torch.cat((sp_features_p3, sp_feature_p3),dim=0)
    #                 sp_features_p4 = torch.cat((sp_features_p4, sp_feature_p4),dim=0)
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1, 3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature, temp_feature_rgb), axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature, temp_feature_lab), axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature, temp_feature_hsv), axis=0)
    #
    #                 RGB_features = np.concatenate((RGB_features,RGB_feature.transpose()),axis=0)
    #                 LAB_features = np.concatenate((LAB_features,LAB_feature.transpose()),axis=0)
    #                 HSV_features = np.concatenate((HSV_features,HSV_feature.transpose()),axis=0)
    #
    #         RGB_features = torch.tensor(RGB_features)
    #         LAB_features = torch.tensor(LAB_features)
    #         HSV_features = torch.tensor(HSV_features)
    #
    #         # make similarity
    #         aff_sp_f = torch.nn.functional.cosine_similarity(sp_features_p1.unsqueeze(0), sp_features_p1.unsqueeze(1), dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p2.unsqueeze(0), sp_features_p2.unsqueeze(1),dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p3.unsqueeze(0), sp_features_p3.unsqueeze(1),dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p4.unsqueeze(0), sp_features_p4.unsqueeze(1),dim=-1)
    #         # aff_sp_f = torch.nn.functional.cosine_similarity(sp_features_p4.unsqueeze(0), sp_features_p4.unsqueeze(1),dim=-1)
    #
    #         aff_score = torch.nn.functional.cosine_similarity(sp_scores.unsqueeze(0), sp_scores.unsqueeze(1), dim=-1)
    #
    #         aff_color = torch.nn.functional.cosine_similarity(RGB_features.unsqueeze(0), RGB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(HSV_features.unsqueeze(0), HSV_features.unsqueeze(1), dim=-1)
    #         # aff_color = torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1), dim=-1)
    #
    #         aff_mat = aff_sp_f + aff_color + aff_score
    #         # aff_mat = aff_sp_f * aff_color
    #         # aff_mat = aff_sp_f * aff_score * aff_rgb
    #
    #         aff_mat_mask = aff_mat * (torch.tensor(adjacent_matrix, dtype=torch.float32) + torch.eye(adjacent_matrix.shape[0]))
    #
    #         trans_mat = aff_mat_mask / (torch.sum(aff_mat_mask, dim=0, keepdim=True)+1e-5)
    #
    #         plt.close()
    #         fig,axx = plt.subplots(2, 3)
    #
    #         sns.heatmap(aff_sp_f.numpy(),cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,0])
    #         axx[0,0].set_title("aff_sp_f")
    #
    #         sns.heatmap(aff_score.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,1])
    #         axx[0,1].set_title("aff_score")
    #
    #         # axx[0,2].axis("off")
    #         sns.heatmap(aff_color.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,2])
    #         axx[0,2].set_title("aff_color")
    #
    #         sns.heatmap(aff_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,0])
    #         axx[1,0].set_title("aff_mat")
    #
    #         sns.heatmap(aff_mat_mask.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,1])
    #         axx[1,1].set_title("aff_mat_mask")
    #
    #         sns.heatmap(trans_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,2])
    #         axx[1,2].set_title("trans_mat")
    #
    #         print(os.path.join(dir,str(img_id)+"_aff.jpg"))
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+"_aff.jpg"), dpi=300)
    #
    #         # 包括背景类 --> (21, sp_num)
    #         RW_cams = torch.tensor(sp_scores_normalize.t(), dtype=torch.float32)
    #         trans_mat = torch.tensor(trans_mat, dtype=torch.float32)
    #
    #         cam_valid_peak_list, cam_peak_score = self.get_peak_list(class_response_maps,lables,peak_threshold)
    #         response_maps = [class_response_maps]
    #         valid_peak_list = [cam_valid_peak_list]
    #         scale = 0.7
    #
    #         # RW_cams = torch.matmul(torch.tensor(trans_mat.t(),dtype=torch.float32),torch.tensor(sp_cams,dtype=torch.float32))
    #         # RW_cams = RW_cams.t()
    #
    #         iter_TIME = 4 + 1
    #         alph = 1
    #
    #         for i in range(iter_TIME):
    #             # print(RW_cams.shape)
    #             # print(sp_scores_normalize.shape)
    #             if i == 0:
    #                 RW_cams = RW_cams
    #             else:
    #                 RW_cams = alph * torch.matmul(RW_cams, trans_mat) + (1 - alph) * torch.tensor(sp_scores_normalize.t(),dtype=torch.float32)
    #             RW_cams = torch.nn.functional.normalize(RW_cams, p=1, dim=0)
    #
    #             RW_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_cams.dtype)
    #
    #             for cls_idx in lables:
    #                 for idx, temp_label in enumerate(range(low_label, high_label + 1)):
    #                     sp_mask = sp_label == temp_label
    #                     sp_mask_tensor = torch.tensor(sp_mask)
    #                     # exclusive bg class
    #                     RW_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_cams[cls_idx + 1,idx] # single batch input
    #
    #             RW_class_response_maps = F.upsample(RW_class_response_maps,size=(cam_H,cam_W),mode='bilinear', align_corners=True)
    #
    #             RW_valid_peak_list, RW_peak_score = self.get_peak_list(RW_class_response_maps,lables,0,self._lzc_filter)
    #
    #             response_maps.append(RW_class_response_maps)
    #             valid_peak_list.append(RW_valid_peak_list)
    #
    #         RW_class_response_maps = RW_class_response_maps.numpy()
    #         RW_class_response_maps = RW_class_response_maps - np.min(RW_class_response_maps,(-2,-1),keepdims = True)
    #         RW_class_response_maps = RW_class_response_maps / np.max(RW_class_response_maps,(-2,-1),keepdims = True)
    #         RW_class_response_maps = torch.tensor(RW_class_response_maps)
    #
    #         # print(RW_class_response_maps)
    #         mask_CAM = RW_class_response_maps * class_response_maps
    #         mask_CAM_valid_peak_list, mask_CAM_peak_score = self.get_peak_list(mask_CAM, lables, peak_threshold)
    #         response_maps.append(mask_CAM)
    #         valid_peak_list.append(mask_CAM_valid_peak_list)
    #
    #         cam_num = onehot_label.sum().item()
    #         plt.close()
    #         f, axarr = plt.subplots(len(response_maps), int(cam_num + 1))
    #         plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.1, hspace=0.1)
    #
    #         for idx in range(len(response_maps)):
    #             if idx == 0:
    #                 axarr[idx, 0].imshow(img)
    #             else:
    #                 axarr[idx, 0].imshow(mark_boundaries(img, sp_label))
    #
    #             axarr[idx, 0].axis("off")
    #
    #             for i,cls_idx in enumerate(lables):
    #                 CRM = response_maps[idx]
    #
    #                 cam = CRM[0, cls_idx].detach().cpu().numpy()
    #                 cam = cam - np.min(cam)
    #                 cam = cam / np.max(cam)
    #                 cam = np.uint8(255 * cam)
    #
    #                 heatmap = cv2.applyColorMap(cv2.resize(cam, (W, H)), cv2.COLORMAP_JET)
    #                 heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #                 result = heatmap * scale + (img) * (1 - scale)
    #                 result = np.uint8(result)
    #                 result = cv2.resize(result, (int(cam_W*self.sub_pixel_locating_factor),int(cam_H*self.sub_pixel_locating_factor)))
    #
    #                 axarr[idx, i+1].imshow(result)
    #                 axarr[idx, i+1].axis("off")
    #                 axarr[idx, i+1].set_title(voc_id_name_map[cls_idx.item()+1])
    #
    #                 VPL = valid_peak_list[idx]
    #
    #                 for peak in VPL:
    #                     if peak[1] == cls_idx:
    #                         axarr[idx, i+1].scatter(peak[3],peak[2], color='purple',marker='*',edgecolors='purple', s=17)
    #
    #                 axarr[idx, i+1].axis("off")
    #
    #         # plt.show()
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+".jpg"), dpi=300)
    #
    #         # if len(peak_score) > 0:
    #         #     valid_peak_list = torch.stack(valid_peak_list)
    #         #     peak_score = torch.tensor(peak_score)
    #         #     # classification confidence scores, class-aware and instance-aware visual cues
    #         #     return class_response_maps, valid_peak_list, peak_score
    #         # else:
    #         #     return None

    # # use proposal score to replace cam score
    # # label assign
    # def forward(self, input, lables, img,class_threshold=0, peak_threshold=10, retrieval_cfg=None,graph_op=None,resolution=7):
    #     assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
    #     with torch.no_grad():
    #         device = input.device
    #
    #         # _, _, H, W = input.shape  # batchsize == 1
    #         H,W,_ = np.array(img).shape
    #
    #         LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #         HSV_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #
    #         # classification network forwarding
    #         class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
    #         class_response_maps = class_response_maps.detach()
    #         org_size_class_response_maps = F.upsample(class_response_maps, size=(H,W), mode='bilinear', align_corners=True)
    #
    #         _, cls_num, cam_H, cam_W = class_response_maps.shape
    #
    #         onehot_label = self.label2onehot(lables,cls_num)
    #
    #         _, _, f_H_p1, f_W_p1 = p1.shape
    #         _, _, f_H_p2, f_W_p2 = p2.shape
    #         _, _, f_H_p3, f_W_p3 = p3.shape
    #         _, _, f_H_p4, f_W_p4 = p4.shape
    #
    #         Align_op_p1 = RoIAlign((resolution, resolution), f_H_p1/H, 0)
    #         Align_op_p2 = RoIAlign((resolution, resolution), f_H_p2/H, 0)
    #         Align_op_p3 = RoIAlign((resolution, resolution), f_H_p3/H, 0)
    #         Align_op_p4 = RoIAlign((resolution, resolution), f_H_p4/H, 0)
    #
    #         # graph operate
    #         sp_label = graph_op["sp_label"]
    #         adjacent_matrix = graph_op["adjacent_matrix"]
    #         img_id = graph_op["img_id"]
    #         belong_mat = graph_op["belong_mat"]
    #         proposal_score = graph_op["proposal_score"]
    #
    #         # add bg_score
    #         bg_score = 1 - proposal_score.sum(-1)
    #         proposal_score = np.concatenate((bg_score[:,None],proposal_score),axis=1)
    #
    #         sp_scores = np.zeros((adjacent_matrix.shape[0],proposal_score.shape[-1])) # sp_num, cls_num + 1
    #         for row_idx in range(belong_mat.shape[0]):
    #             score = proposal_score[row_idx]
    #             belong_row = belong_mat[row_idx] != 0
    #
    #             sp_scores[belong_row] += score
    #
    #         sp_scores = torch.tensor(sp_scores).to(device)
    #         sp_scores_normalize = torch.nn.functional.normalize(sp_scores,p=1,dim=-1)
    #
    #         low_label = np.min(sp_label)
    #         high_label = np.max(sp_label)
    #
    #
    #         for idx, temp_label in enumerate(range(low_label, high_label + 1)):
    #             sp_mask = sp_label == temp_label
    #
    #             # sp_mask_tensor = torch.tensor(sp_mask)
    #
    #             ind_xy = np.nonzero(sp_mask)
    #             xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
    #
    #             mask = sp_mask[ymin:ymax, xmin:xmax]
    #             mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
    #             mask = torch.tensor(mask).to(device)
    #
    #             box = torch.tensor([[0, xmin, ymin, xmax, ymax]],dtype=torch.float32).to(device)
    #             box_feature_p1 = Align_op_p1(p1, box)  # (1,7,7)
    #             box_feature_p2 = Align_op_p2(p2, box)  # (1,7,7)
    #             box_feature_p3 = Align_op_p3(p3, box)  # (1,7,7)
    #             box_feature_p4 = Align_op_p4(p4, box)  # (1,7,7)
    #
    #             sp_feature_p1 = get_sp_f(box_feature_p1,mask,resolution)
    #             sp_feature_p2 = get_sp_f(box_feature_p2,mask,resolution)
    #             sp_feature_p3 = get_sp_f(box_feature_p3,mask,resolution)
    #             sp_feature_p4 = get_sp_f(box_feature_p4,mask,resolution)
    #
    #             if idx == 0:
    #                 sp_features_p1 = sp_feature_p1
    #                 sp_features_p2 = sp_feature_p2
    #                 sp_features_p3 = sp_feature_p3
    #                 sp_features_p4 = sp_feature_p4
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1,3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature,temp_feature_rgb),axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature,temp_feature_lab),axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature,temp_feature_hsv),axis=0)
    #
    #                 RGB_features = RGB_feature.transpose()
    #                 LAB_features = LAB_feature.transpose()
    #                 HSV_features = HSV_feature.transpose()
    #
    #             else:
    #                 sp_features_p1 = torch.cat((sp_features_p1, sp_feature_p1),dim=0)
    #                 sp_features_p2 = torch.cat((sp_features_p2, sp_feature_p2),dim=0)
    #                 sp_features_p3 = torch.cat((sp_features_p3, sp_feature_p3),dim=0)
    #                 sp_features_p4 = torch.cat((sp_features_p4, sp_feature_p4),dim=0)
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1, 3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature, temp_feature_rgb), axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature, temp_feature_lab), axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature, temp_feature_hsv), axis=0)
    #
    #                 RGB_features = np.concatenate((RGB_features,RGB_feature.transpose()),axis=0)
    #                 LAB_features = np.concatenate((LAB_features,LAB_feature.transpose()),axis=0)
    #                 HSV_features = np.concatenate((HSV_features,HSV_feature.transpose()),axis=0)
    #
    #         RGB_features = torch.tensor(RGB_features).to(device)
    #         LAB_features = torch.tensor(LAB_features).to(device)
    #         HSV_features = torch.tensor(HSV_features).to(device)
    #
    #         # make similarity
    #         aff_sp_f = torch.nn.functional.cosine_similarity(sp_features_p1.unsqueeze(0), sp_features_p1.unsqueeze(1), dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p2.unsqueeze(0), sp_features_p2.unsqueeze(1),dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p3.unsqueeze(0), sp_features_p3.unsqueeze(1),dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p4.unsqueeze(0), sp_features_p4.unsqueeze(1),dim=-1)
    #
    #         aff_score = torch.nn.functional.cosine_similarity(sp_scores.unsqueeze(0), sp_scores.unsqueeze(1), dim=-1)
    #
    #         aff_color = torch.nn.functional.cosine_similarity(RGB_features.unsqueeze(0), RGB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(HSV_features.unsqueeze(0), HSV_features.unsqueeze(1), dim=-1)
    #
    #         aff_mat = aff_sp_f.to(device) + aff_color.to(device) + aff_score.to(device)
    #
    #         aff_mat_mask = aff_mat * (torch.tensor(adjacent_matrix, dtype=torch.float32) +
    #                                   torch.eye(adjacent_matrix.shape[0])).to(device)
    #
    #         trans_mat = aff_mat_mask / (torch.sum(aff_mat_mask, dim=0, keepdim=True)+1e-5)
    #
    #
    #         # 包括背景类 --> (21, sp_num)
    #         RW_cams = torch.tensor(sp_scores_normalize.t(), dtype=torch.float32)
    #         trans_mat = torch.tensor(trans_mat, dtype=torch.float32)
    #
    #         iter_TIME = 4
    #         alph = 1
    #
    #         for i in range(iter_TIME):
    #             if i == 0:
    #                 RW_cams = RW_cams
    #             else:
    #                 RW_cams = alph * torch.matmul(RW_cams, trans_mat)
    #             RW_cams = torch.nn.functional.normalize(RW_cams, p=1, dim=0)
    #
    #             if i+1 == iter_TIME:
    #                 RW_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_cams.dtype)
    #
    #                 for cls_idx in lables:
    #                     for idx, temp_label in enumerate(range(low_label, high_label + 1)):
    #                         sp_mask = sp_label == temp_label
    #                         sp_mask_tensor = torch.tensor(sp_mask)
    #                         # exclusive bg class
    #                         RW_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_cams[cls_idx + 1,idx] # single batch input
    #
    #                 RW_class_response_maps = F.upsample(RW_class_response_maps,size=(cam_H,cam_W),mode='bilinear', align_corners=True)
    #
    #         RW_class_response_maps = RW_class_response_maps.cpu().numpy()
    #         RW_class_response_maps = RW_class_response_maps - np.min(RW_class_response_maps,(-2,-1),keepdims = True)
    #         RW_class_response_maps = RW_class_response_maps / (1e-5 + np.max(RW_class_response_maps,(-2,-1),keepdims = True))
    #         RW_class_response_maps = torch.tensor(RW_class_response_maps).to(device)
    #
    #         mask_CAM = RW_class_response_maps * class_response_maps
    #         mask_CAM_valid_peak_list, mask_CAM_peak_score = self.get_peak_list(mask_CAM, lables, peak_threshold)
    #
    #         if len(mask_CAM_valid_peak_list) > 0:
    #             valid_peak_list = torch.stack(mask_CAM_valid_peak_list)
    #             peak_score = torch.tensor(mask_CAM_peak_score)
    #             return mask_CAM, valid_peak_list, peak_score
    #         else:
    #             return None

    # use proposal score to replace cam score v2
    # label assign
    def forward(self, input, lables, img, class_threshold=0, peak_threshold=10, retrieval_cfg=None, graph_op=None,
                resolution=7):
        assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
        with torch.no_grad():
            device = input.device

            # _, _, H, W = input.shape  # batchsize == 1
            H, W, _ = np.array(img).shape

            LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            HSV_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # classification network forwarding
            class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
            class_response_maps = class_response_maps.detach()
            org_size_class_response_maps = F.upsample(class_response_maps, size=(H, W), mode='bilinear',
                                                      align_corners=True)

            _, cls_num, cam_H, cam_W = class_response_maps.shape

            onehot_label = self.label2onehot(lables, cls_num)

            _, _, f_H_p1, f_W_p1 = p1.shape
            _, _, f_H_p2, f_W_p2 = p2.shape
            _, _, f_H_p3, f_W_p3 = p3.shape
            _, _, f_H_p4, f_W_p4 = p4.shape

            Align_op_p1 = RoIAlign((resolution, resolution), f_H_p1 / H, 0)
            Align_op_p2 = RoIAlign((resolution, resolution), f_H_p2 / H, 0)
            Align_op_p3 = RoIAlign((resolution, resolution), f_H_p3 / H, 0)
            Align_op_p4 = RoIAlign((resolution, resolution), f_H_p4 / H, 0)

            # graph operate
            sp_label = graph_op["sp_label"]
            adjacent_matrix = graph_op["adjacent_matrix"]
            img_id = graph_op["img_id"]
            belong_mat = graph_op["belong_mat"]
            proposal_score = graph_op["proposal_score"]

            # add bg_score
            bg_score = 1 - proposal_score.sum(-1)
            proposal_score = np.concatenate((bg_score[:, None], proposal_score), axis=1)

            # lzc edit 9-6
            max_idx = np.argmax(proposal_score, axis=-1)
            flag = np.zeros_like(max_idx)
            for label in lables:
                flag += (label + 1).item() == max_idx

            bg_score_const = np.zeros(proposal_score.shape[-1])
            bg_score_const[0] = 1
            proposal_score[flag == 0] = bg_score_const

            sp_scores = np.zeros((adjacent_matrix.shape[0], proposal_score.shape[-1]))  # sp_num, cls_num + 1
            for row_idx in range(belong_mat.shape[0]):
                score = proposal_score[row_idx]
                belong_row = belong_mat[row_idx] != 0

                sp_scores[belong_row] += score

            sp_scores = torch.tensor(sp_scores).to(device)
            sp_scores_normalize = torch.nn.functional.normalize(sp_scores, p=1, dim=-1)

            low_label = np.min(sp_label)
            high_label = np.max(sp_label)

            for idx, temp_label in enumerate(range(low_label, high_label + 1)):
                sp_mask = sp_label == temp_label

                # sp_mask_tensor = torch.tensor(sp_mask)

                ind_xy = np.nonzero(sp_mask)
                xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1

                mask = sp_mask[ymin:ymax, xmin:xmax]
                mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
                mask = torch.tensor(mask).to(device)

                box = torch.tensor([[0, xmin, ymin, xmax, ymax]], dtype=torch.float32).to(device)
                box_feature_p1 = Align_op_p1(p1, box)  # (1,7,7)
                box_feature_p2 = Align_op_p2(p2, box)  # (1,7,7)
                box_feature_p3 = Align_op_p3(p3, box)  # (1,7,7)
                box_feature_p4 = Align_op_p4(p4, box)  # (1,7,7)

                sp_feature_p1 = get_sp_f(box_feature_p1, mask, resolution)
                sp_feature_p2 = get_sp_f(box_feature_p2, mask, resolution)
                sp_feature_p3 = get_sp_f(box_feature_p3, mask, resolution)
                sp_feature_p4 = get_sp_f(box_feature_p4, mask, resolution)

                if idx == 0:
                    sp_features_p1 = sp_feature_p1
                    sp_features_p2 = sp_feature_p2
                    sp_features_p3 = sp_feature_p3
                    sp_features_p4 = sp_feature_p4

                    RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
                    LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
                    HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
                    for color_idx in range(1, 3):
                        temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
                        temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
                        temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])

                        RGB_feature = np.concatenate((RGB_feature, temp_feature_rgb), axis=0)
                        LAB_feature = np.concatenate((LAB_feature, temp_feature_lab), axis=0)
                        HSV_feature = np.concatenate((HSV_feature, temp_feature_hsv), axis=0)

                    RGB_features = RGB_feature.transpose()
                    LAB_features = LAB_feature.transpose()
                    HSV_features = HSV_feature.transpose()

                else:
                    sp_features_p1 = torch.cat((sp_features_p1, sp_feature_p1), dim=0)
                    sp_features_p2 = torch.cat((sp_features_p2, sp_feature_p2), dim=0)
                    sp_features_p3 = torch.cat((sp_features_p3, sp_feature_p3), dim=0)
                    sp_features_p4 = torch.cat((sp_features_p4, sp_feature_p4), dim=0)

                    RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
                    LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
                    HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
                    for color_idx in range(1, 3):
                        temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
                        temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
                        temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])

                        RGB_feature = np.concatenate((RGB_feature, temp_feature_rgb), axis=0)
                        LAB_feature = np.concatenate((LAB_feature, temp_feature_lab), axis=0)
                        HSV_feature = np.concatenate((HSV_feature, temp_feature_hsv), axis=0)

                    RGB_features = np.concatenate((RGB_features, RGB_feature.transpose()), axis=0)
                    LAB_features = np.concatenate((LAB_features, LAB_feature.transpose()), axis=0)
                    HSV_features = np.concatenate((HSV_features, HSV_feature.transpose()), axis=0)

            RGB_features = torch.tensor(RGB_features).to(device)
            LAB_features = torch.tensor(LAB_features).to(device)
            HSV_features = torch.tensor(HSV_features).to(device)

            # make similarity
            aff_sp_f = torch.nn.functional.cosine_similarity(sp_features_p1.unsqueeze(0), sp_features_p1.unsqueeze(1),
                                                             dim=-1) * \
                       torch.nn.functional.cosine_similarity(sp_features_p2.unsqueeze(0), sp_features_p2.unsqueeze(1),
                                                             dim=-1) * \
                       torch.nn.functional.cosine_similarity(sp_features_p3.unsqueeze(0), sp_features_p3.unsqueeze(1),
                                                             dim=-1) * \
                       torch.nn.functional.cosine_similarity(sp_features_p4.unsqueeze(0), sp_features_p4.unsqueeze(1),
                                                             dim=-1)

            aff_score = torch.nn.functional.cosine_similarity(sp_scores.unsqueeze(0), sp_scores.unsqueeze(1), dim=-1)

            aff_color = torch.nn.functional.cosine_similarity(RGB_features.unsqueeze(0), RGB_features.unsqueeze(1),
                                                              dim=-1) * \
                        torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1),
                                                              dim=-1) * \
                        torch.nn.functional.cosine_similarity(HSV_features.unsqueeze(0), HSV_features.unsqueeze(1),
                                                              dim=-1)

            aff_mat = aff_sp_f.to(device) + aff_color.to(device) + aff_score.to(device)

            aff_mat_mask = aff_mat * (torch.tensor(adjacent_matrix, dtype=torch.float32) +
                                      torch.eye(adjacent_matrix.shape[0])).to(device)

            trans_mat = aff_mat_mask / (torch.sum(aff_mat_mask, dim=0, keepdim=True) + 1e-5)

            # 包括背景类 --> (21, sp_num)
            RW_cams = torch.tensor(sp_scores_normalize.t(), dtype=torch.float32)
            trans_mat = torch.tensor(trans_mat, dtype=torch.float32)

            iter_TIME = 4

            for i in range(iter_TIME):
                if i == 0:
                    RW_cams = RW_cams
                else:
                    RW_cams = torch.matmul(RW_cams, trans_mat)
                RW_cams = torch.nn.functional.normalize(RW_cams, p=1, dim=0)

                if i + 1 == iter_TIME:
                    RW_class_response_maps = torch.zeros_like(org_size_class_response_maps, dtype=RW_cams.dtype)

                    for cls_idx in lables:
                        for idx, temp_label in enumerate(range(low_label, high_label + 1)):
                            sp_mask = sp_label == temp_label
                            sp_mask_tensor = torch.tensor(sp_mask)
                            # exclusive bg class
                            RW_class_response_maps[0, cls_idx, sp_mask_tensor] = RW_cams[
                                cls_idx + 1, idx]  # single batch input

                    RW_class_response_maps = F.upsample(RW_class_response_maps, size=(cam_H, cam_W), mode='bilinear',
                                                        align_corners=True)

            RW_class_response_maps = RW_class_response_maps.cpu().numpy()
            RW_class_response_maps = RW_class_response_maps - np.min(RW_class_response_maps, (-2, -1), keepdims=True)
            RW_class_response_maps = RW_class_response_maps / (
                        1e-5 + np.max(RW_class_response_maps, (-2, -1), keepdims=True))
            RW_class_response_maps = torch.tensor(RW_class_response_maps).to(device)

            # RW_valid_peak_list, RW_peak_score = self.get_peak_list(RW_class_response_maps,lables,0.5)

            mask_CAM = RW_class_response_maps * class_response_maps
            mask_CAM_valid_peak_list, mask_CAM_peak_score = self.get_peak_list(mask_CAM, lables, peak_threshold)

            if len(mask_CAM_valid_peak_list) > 0:
                valid_peak_list = torch.stack(mask_CAM_valid_peak_list)
                peak_score = torch.tensor(mask_CAM_peak_score)
                return RW_class_response_maps, valid_peak_list, peak_score
            else:
                return None

    # # edit 9.4 lzc
    # def forward(self, input, lables, img,class_threshold=0, peak_threshold=10, retrieval_cfg=None,graph_op=None,resolution=7):
    #     assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
    #     dir = "./voc_visual"
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #     with torch.no_grad():
    #
    #         # _, _, H, W = input.shape  # batchsize == 1
    #         H,W,_ = np.array(img).shape
    #
    #         LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #         HSV_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #
    #         # classification network forwarding
    #         class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
    #
    #         _, cls_num, cam_H, cam_W = class_response_maps.shape
    #
    #         onehot_label = torch.tensor(self.label2onehot(lables,cls_num))
    #         class_response_maps = class_response_maps * torch.tensor(onehot_label).view(1,cls_num,1,1)
    #
    #         class_response_maps = class_response_maps.detach()
    #         org_size_class_response_maps = F.upsample(class_response_maps, size=(H,W), mode='bilinear', align_corners=True)
    #
    #         # use p2 as feature
    #         feature = p4
    #
    #         _, _, f_H, f_W = feature.shape
    #
    #         # spatial_scale
    #         spatial_scale = f_H / H
    #         Align_op = RoIAlign((resolution, resolution), spatial_scale, 0)
    #
    #         # graph operate
    #         sp_label = graph_op["sp_label"]
    #         adjacent_matrix = graph_op["adjacent_matrix"]
    #         img_id = graph_op["img_id"]
    #         belong_mat = graph_op["belong_mat"]
    #         proposal_score = graph_op["proposal_score"]
    #         print(img_id)
    #
    #         # add bg_score
    #         bg_score = 1 - proposal_score.sum(-1)
    #         proposal_score = np.concatenate((bg_score[:,None],proposal_score),axis=1)
    #
    #         sp_scores = np.zeros((adjacent_matrix.shape[0],proposal_score.shape[-1])) # sp_num, cls_num + 1
    #         for row_idx in range(belong_mat.shape[0]):
    #             score = proposal_score[row_idx]
    #             belong_row = belong_mat[row_idx] != 0
    #
    #             sp_scores[belong_row] += score
    #
    #         sp_scores = torch.tensor(sp_scores)
    #         # sp_scores_normalize = torch.nn.functional.normalize(sp_scores,p=1,dim=-1)
    #
    #         low_label = np.min(sp_label)
    #         high_label = np.max(sp_label)
    #
    #         for idx, temp_label in tqdm(enumerate(range(low_label, high_label + 1)),total=adjacent_matrix.shape[0]):
    #             sp_mask = sp_label == temp_label
    #
    #             sp_mask_tensor = torch.tensor(sp_mask)
    #             cam_scores = (sp_mask_tensor * org_size_class_response_maps).sum((-2,-1)) / sp_mask_tensor.sum() # --> (1, cls_num)
    #
    #             ind_xy = np.nonzero(sp_mask)
    #             xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
    #
    #             mask = sp_mask[ymin:ymax, xmin:xmax]
    #             mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
    #             mask = torch.tensor(mask)
    #
    #             box = torch.tensor([[0, xmin, ymin, xmax, ymax]],dtype=torch.float32)
    #             box_feature = Align_op(feature, box)  # (1,7,7)
    #
    #             sp_f = (mask * box_feature).view(-1, resolution, resolution)  # (1,dim,7,7) --> (dim,7,7)
    #             sp_f = sp_f.sum((-2,-1))  # (dim)
    #             sp_f = sp_f / (mask.sum()+1e-5)
    #
    #             sp_f = sp_f[None, :]
    #
    #             if idx == 0:
    #                 sp_cams = cam_scores
    #                 sp_features = sp_f
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1,3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature,temp_feature_rgb),axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature,temp_feature_lab),axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature,temp_feature_hsv),axis=0)
    #
    #                 RGB_features = RGB_feature.transpose()
    #                 LAB_features = LAB_feature.transpose()
    #                 HSV_features = HSV_feature.transpose()
    #
    #             else:
    #                 sp_cams = torch.cat((sp_cams, cam_scores), dim=0)
    #                 sp_features = torch.cat((sp_features, sp_f), dim=0)
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1, 3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature, temp_feature_rgb), axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature, temp_feature_lab), axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature, temp_feature_hsv), axis=0)
    #
    #                 RGB_features = np.concatenate((RGB_features,RGB_feature.transpose()),axis=0)
    #                 LAB_features = np.concatenate((LAB_features,LAB_feature.transpose()),axis=0)
    #                 HSV_features = np.concatenate((HSV_features,HSV_feature.transpose()),axis=0)
    #
    #         RGB_features = torch.tensor(RGB_features)
    #         LAB_features = torch.tensor(LAB_features)
    #         HSV_features = torch.tensor(HSV_features)
    #
    #
    #         # make similarity
    #         aff_sp_f = torch.nn.functional.cosine_similarity(sp_features.unsqueeze(0), sp_features.unsqueeze(1), dim=-1)
    #         aff_score = torch.nn.functional.cosine_similarity(sp_scores.unsqueeze(0), sp_scores.unsqueeze(1), dim=-1)
    #         aff_color = torch.nn.functional.cosine_similarity(RGB_features.unsqueeze(0), RGB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(HSV_features.unsqueeze(0), HSV_features.unsqueeze(1), dim=-1)
    #
    #
    #         aff_mat = aff_sp_f + aff_score + aff_color
    #         # aff_mat = aff_sp_f * aff_score * aff_rgb
    #
    #         aff_mat_mask = aff_mat * (torch.tensor(adjacent_matrix, dtype=torch.float32) + torch.eye(adjacent_matrix.shape[0]))
    #
    #         trans_mat = aff_mat_mask / (torch.sum(aff_mat_mask, dim=0, keepdim=True)+1e-5)
    #
    #         # logt = 4
    #         #
    #         # for _ in range(logt):
    #         #     trans_mat = torch.matmul(trans_mat, trans_mat)
    #
    #         # trans_mat = torch.linalg.inv(1e-5 * torch.eye(aff_mat.shape[0]) + torch.tensor(aff_mat_mask,dtype=torch.float32)) @ torch.tensor(aff_mat_mask,dtype=torch.float32)
    #
    #         plt.close()
    #         fig,axx = plt.subplots(2, 3)
    #
    #         sns.heatmap(aff_sp_f.numpy(),cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,0])
    #         axx[0,0].set_title("aff_sp_f")
    #
    #         sns.heatmap(aff_score.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,1])
    #         axx[0,1].set_title("aff_score")
    #
    #         # axx[0,2].axis("off")
    #         sns.heatmap(aff_color.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,2])
    #         axx[0,2].set_title("aff_color")
    #
    #         sns.heatmap(aff_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,0])
    #         axx[1,0].set_title("aff_mat")
    #
    #         sns.heatmap(aff_mat_mask.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,1])
    #         axx[1,1].set_title("aff_mat_mask")
    #
    #         sns.heatmap(trans_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,2])
    #         axx[1,2].set_title("trans_mat")
    #
    #         print(os.path.join(dir,str(img_id)+"_aff.jpg"))
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+"_aff.jpg"), dpi=300)
    #
    #         RW_cams = torch.tensor(sp_cams.t(), dtype=torch.float32)
    #         trans_mat = torch.tensor(trans_mat, dtype=torch.float32)
    #
    #         cam_valid_peak_list, cam_peak_score = self.get_peak_list(class_response_maps,lables,peak_threshold)
    #         response_maps = [class_response_maps]
    #         valid_peak_list = [cam_valid_peak_list]
    #         scale = 0.7
    #
    #         # RW_cams = torch.matmul(torch.tensor(trans_mat.t(),dtype=torch.float32),torch.tensor(sp_cams,dtype=torch.float32))
    #         # RW_cams = RW_cams.t()
    #
    #         iter_TIME = 4
    #         alph = 0.7
    #
    #         for _ in range(iter_TIME):
    #             RW_cams = alph * torch.matmul(RW_cams, trans_mat) + (1 - alph) * torch.tensor(sp_cams.t(),dtype=torch.float32)
    #             RW_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_cams.dtype)
    #
    #             for cls_idx in lables:
    #                 for idx, temp_label in enumerate(range(low_label, high_label + 1)):
    #                     sp_mask = sp_label == temp_label
    #                     sp_mask_tensor = torch.tensor(sp_mask)
    #                     RW_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_cams[cls_idx,temp_label] # single batch input
    #
    #             RW_class_response_maps = F.upsample(RW_class_response_maps,size=(cam_H,cam_W),mode='bilinear', align_corners=True)
    #
    #             RW_valid_peak_list, RW_peak_score = self.get_peak_list(RW_class_response_maps,lables,peak_threshold)
    #
    #             response_maps.append(RW_class_response_maps)
    #             valid_peak_list.append(RW_valid_peak_list)
    #
    #         cam_num = onehot_label.sum().item()
    #         plt.close()
    #         f, axarr = plt.subplots(len(response_maps), int(cam_num + 1))
    #         plt.tight_layout()
    #
    #         for idx in range(len(response_maps)):
    #             if idx == 0:
    #                 axarr[idx, 0].imshow(img)
    #             else:
    #                 axarr[idx, 0].imshow(mark_boundaries(img, sp_label))
    #
    #             axarr[idx, 0].axis("off")
    #
    #             for i,cls_idx in enumerate(lables):
    #                 CRM = response_maps[idx]
    #
    #                 cam = CRM[0, cls_idx].detach().cpu().numpy()
    #                 cam = cam - np.min(cam)
    #                 cam = cam / np.max(cam)
    #                 cam = np.uint8(255 * cam)
    #
    #                 heatmap = cv2.applyColorMap(cv2.resize(cam, (W, H)), cv2.COLORMAP_JET)
    #                 heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #                 result = heatmap * scale + (img) * (1 - scale)
    #                 result = np.uint8(result)
    #                 result = cv2.resize(result, (int(cam_W*self.sub_pixel_locating_factor),int(cam_H*self.sub_pixel_locating_factor)))
    #
    #                 axarr[idx, i+1].imshow(result)
    #                 axarr[idx, i+1].axis("off")
    #                 axarr[idx, i+1].set_title(voc_id_name_map[cls_idx.item()+1])
    #
    #                 VPL = valid_peak_list[idx]
    #
    #                 for peak in VPL:
    #                     if peak[1] == cls_idx:
    #                         axarr[idx, i+1].scatter(peak[3],peak[2], color='purple',marker='*',edgecolors='purple', s=17)
    #
    #                 axarr[idx, i+1].axis("off")
    #
    #         # plt.show()
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+".jpg"), dpi=300)
    #
    #         # if len(peak_score) > 0:
    #         #     valid_peak_list = torch.stack(valid_peak_list)
    #         #     peak_score = torch.tensor(peak_score)
    #         #     # classification confidence scores, class-aware and instance-aware visual cues
    #         #     return class_response_maps, valid_peak_list, peak_score
    #         # else:
    #         #     return None

    # def forward(self, input, lables, img, class_threshold=0, peak_threshold=10, retrieval_cfg=None, graph_op=None,resolution=7):
    #
    #     assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
    #     with torch.no_grad():
    #
    #         _, _, H, W = input.shape  # batchsize == 1
    #
    #         # classification network forwarding
    #         class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
    #
    #         _, cls_num, cam_H, cam_W = class_response_maps.shape
    #
    #         onehot_label = torch.tensor(self.label2onehot(lables,cls_num))
    #         class_response_maps = class_response_maps * torch.tensor(onehot_label).view(1,cls_num,1,1)
    #
    #         # gap_cam = torch.nn.AdaptiveMaxPool2d((1,1))(class_response_maps)
    #
    #         class_response_maps = class_response_maps.detach()
    #         orig_img_size = img.shape[:2]
    #
    #         class_response_maps_normalize = F.upsample(class_response_maps, orig_img_size, mode='bilinear',align_corners=False)
    #         class_response_maps_normalize = class_response_maps_normalize.numpy()[0]
    #         class_response_maps_normalize = class_response_maps_normalize - np.min(class_response_maps_normalize,axis=(-2,-1))[:, None,None]
    #         class_response_maps_normalize = class_response_maps_normalize / np.max(1e-5 + class_response_maps_normalize,axis=(-2,-1))[:,None, None]
    #         bg_score = np.power(1 - np.max(class_response_maps_normalize, axis=0, keepdims=True), 1)
    #         bgcam_score = np.concatenate((bg_score, class_response_maps_normalize[lables].reshape(len(lables),
    #                                                                                               class_response_maps_normalize.shape[1],
    #                                                                                               class_response_maps_normalize.shape[2])), axis=0)
    #         crf_score = crf_inference(img, bgcam_score, t=1, labels=bgcam_score.shape[0])
    #         crf_score = torch.tensor(crf_score)[None, 1:]
    #         crf_score = F.upsample(crf_score, size=(cam_H, cam_W), mode='bilinear', align_corners=True)
    #
    #         cam_valid_peak_list, cam_peak_score = self.get_peak_list(class_response_maps,lables,peak_threshold)
    #         crf_valid_peak_list, crf_peak_score = self.get_peak_list(crf_score,torch.arange(len(lables)),0.5)
    #
    #         cam_num = onehot_label.sum().item()
    #         f, axarr = plt.subplots(2, int(cam_num + 1))
    #         plt.tight_layout()
    #
    #         response_maps = [class_response_maps,crf_score]
    #         valid_peak_list = [cam_valid_peak_list,crf_valid_peak_list]
    #         scale = 0.7
    #         img = cv2.resize(img,(448,448))
    #
    #         img_id = graph_op["img_id"]
    #
    #         # axarr[0, 0].set_title(str(torch.nn.functional.sigmoid(gap_cam).flatten().numpy()[lables]))
    #
    #         for idx in range(2):
    #             axarr[idx, 0].imshow(img)
    #             axarr[idx, 0].axis("off")
    #
    #             for i,cls_idx in enumerate(lables):
    #                 CRM = response_maps[idx]
    #                 if idx == 0:
    #                     cam = CRM[0, cls_idx].detach().cpu().numpy()
    #                 else:
    #                     cam = CRM[0, i].detach().cpu().numpy()
    #
    #                 cam = cam - np.min(cam)
    #                 cam = cam / np.max(cam)
    #                 cam = np.uint8(255 * cam)
    #
    #                 heatmap = cv2.applyColorMap(cv2.resize(cam, (W, H)), cv2.COLORMAP_JET)
    #                 heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #                 result = heatmap * scale + (img) * (1 - scale)
    #                 result = np.uint8(result)
    #                 result = cv2.resize(result, (int(cam_W*self.sub_pixel_locating_factor),int(cam_H*self.sub_pixel_locating_factor)))
    #
    #                 axarr[idx, i+1].imshow(result)
    #                 axarr[idx, i+1].axis("off")
    #                 axarr[idx, i+1].set_title(voc_id_name_map[cls_idx.item()+1])
    #
    #                 VPL = valid_peak_list[idx]
    #
    #                 for peak in VPL:
    #                     if idx == 0:
    #                         if peak[1] == cls_idx:
    #                             axarr[idx, i+1].scatter(peak[3],peak[2], color='purple',marker='*',edgecolors='purple', s=17)
    #                     else:
    #                         if peak[1] == i:
    #                             axarr[idx, i + 1].scatter(peak[3], peak[2], color='purple', marker='*',
    #                                                       edgecolors='purple', s=17)
    #
    #                 axarr[idx, i+1].axis("off")
    #
    #         dir = "./voc_visual"
    #         if not os.path.exists(dir):
    #             os.makedirs(dir)
    #         plt.savefig(os.path.join(dir,str(img_id)+".jpg"), dpi=300)

    # # cam score and proposal score
    # def forward(self, input, lables, img, class_threshold=0, peak_threshold=10, retrieval_cfg=None, graph_op=None,resolution=7):
    #     assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
    #     dir = "./voc_visual"
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #     with torch.no_grad():
    #
    #         # _, _, H, W = input.shape  # batchsize == 1
    #         H,W,_ = np.array(img).shape
    #
    #         LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #         HSV_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #
    #         # classification network forwarding
    #         class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
    #         class_response_maps = class_response_maps.detach()
    #         org_size_class_response_maps = F.upsample(class_response_maps, size=(H,W), mode='bilinear', align_corners=True)
    #
    #         _, cls_num, cam_H, cam_W = class_response_maps.shape
    #
    #         onehot_label = torch.tensor(self.label2onehot(lables,cls_num))
    #
    #         _, _, f_H_p1, f_W_p1 = p1.shape
    #         _, _, f_H_p2, f_W_p2 = p2.shape
    #         _, _, f_H_p3, f_W_p3 = p3.shape
    #         _, _, f_H_p4, f_W_p4 = p4.shape
    #
    #         Align_op_p1 = RoIAlign((resolution, resolution), f_H_p1/H, 0)
    #         Align_op_p2 = RoIAlign((resolution, resolution), f_H_p2/H, 0)
    #         Align_op_p3 = RoIAlign((resolution, resolution), f_H_p3/H, 0)
    #         Align_op_p4 = RoIAlign((resolution, resolution), f_H_p4/H, 0)
    #
    #         # graph operate
    #         sp_label = graph_op["sp_label"]
    #         adjacent_matrix = graph_op["adjacent_matrix"]
    #         img_id = graph_op["img_id"]
    #         belong_mat = graph_op["belong_mat"]
    #         proposal_score = graph_op["proposal_score"]
    #         print(img_id)
    #
    #         # add bg_score
    #         bg_score = 1 - proposal_score.sum(-1)
    #         proposal_score = np.concatenate((bg_score[:,None],proposal_score),axis=1)
    #
    #         sp_scores = np.zeros((adjacent_matrix.shape[0],proposal_score.shape[-1])) # sp_num, cls_num + 1
    #         for row_idx in range(belong_mat.shape[0]):
    #             score = proposal_score[row_idx]
    #             belong_row = belong_mat[row_idx] != 0
    #
    #             sp_scores[belong_row] += score
    #
    #         sp_scores = torch.tensor(sp_scores)
    #         sp_scores_normalize = torch.nn.functional.normalize(sp_scores,p=1,dim=-1)
    #
    #         low_label = np.min(sp_label)
    #         high_label = np.max(sp_label)
    #
    #         for idx, temp_label in tqdm(enumerate(range(low_label, high_label + 1)),total=adjacent_matrix.shape[0]):
    #             sp_mask = sp_label == temp_label
    #             if idx == 0:
    #                 sp_proposal = sp_mask[None,:,:]
    #             else:
    #                 sp_proposal = np.concatenate((sp_proposal, sp_mask[None,:,:]),axis=0)
    #
    #             sp_mask_tensor = torch.tensor(sp_mask)
    #             cam_scores = (sp_mask_tensor * org_size_class_response_maps).sum(-1).sum(-1) / sp_mask_tensor.sum() # --> (1, cls_num)
    #
    #             ind_xy = np.nonzero(sp_mask)
    #             xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
    #
    #             mask = sp_mask[ymin:ymax, xmin:xmax]
    #             mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
    #             mask = torch.tensor(mask)
    #
    #             box = torch.tensor([[0, xmin, ymin, xmax, ymax]],dtype=torch.float32)
    #             box_feature_p1 = Align_op_p1(p1, box)  # (1,7,7)
    #             box_feature_p2 = Align_op_p2(p2, box)  # (1,7,7)
    #             box_feature_p3 = Align_op_p3(p3, box)  # (1,7,7)
    #             box_feature_p4 = Align_op_p4(p4, box)  # (1,7,7)
    #
    #             sp_feature_p1 = get_sp_f(box_feature_p1,mask,resolution)
    #             sp_feature_p2 = get_sp_f(box_feature_p2,mask,resolution)
    #             sp_feature_p3 = get_sp_f(box_feature_p3,mask,resolution)
    #             sp_feature_p4 = get_sp_f(box_feature_p4,mask,resolution)
    #
    #             if idx == 0:
    #                 sp_cams = cam_scores
    #             else:
    #                 sp_cams = torch.cat((sp_cams, cam_scores), dim=0)
    #
    #             if idx == 0:
    #                 sp_features_p1 = sp_feature_p1
    #                 sp_features_p2 = sp_feature_p2
    #                 sp_features_p3 = sp_feature_p3
    #                 sp_features_p4 = sp_feature_p4
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1,3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature,temp_feature_rgb),axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature,temp_feature_lab),axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature,temp_feature_hsv),axis=0)
    #
    #                 RGB_features = RGB_feature.transpose()
    #                 LAB_features = LAB_feature.transpose()
    #                 HSV_features = HSV_feature.transpose()
    #
    #             else:
    #                 sp_features_p1 = torch.cat((sp_features_p1, sp_feature_p1),dim=0)
    #                 sp_features_p2 = torch.cat((sp_features_p2, sp_feature_p2),dim=0)
    #                 sp_features_p3 = torch.cat((sp_features_p3, sp_feature_p3),dim=0)
    #                 sp_features_p4 = torch.cat((sp_features_p4, sp_feature_p4),dim=0)
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1, 3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature, temp_feature_rgb), axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature, temp_feature_lab), axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature, temp_feature_hsv), axis=0)
    #
    #                 RGB_features = np.concatenate((RGB_features,RGB_feature.transpose()),axis=0)
    #                 LAB_features = np.concatenate((LAB_features,LAB_feature.transpose()),axis=0)
    #                 HSV_features = np.concatenate((HSV_features,HSV_feature.transpose()),axis=0)
    #
    #         RGB_features = torch.tensor(RGB_features)
    #         LAB_features = torch.tensor(LAB_features)
    #         HSV_features = torch.tensor(HSV_features)
    #
    #
    #         # make similarity
    #         aff_sp_f = torch.nn.functional.cosine_similarity(sp_features_p1.unsqueeze(0), sp_features_p1.unsqueeze(1), dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p2.unsqueeze(0), sp_features_p2.unsqueeze(1),dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p3.unsqueeze(0), sp_features_p3.unsqueeze(1),dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p4.unsqueeze(0), sp_features_p4.unsqueeze(1),dim=-1)
    #
    #         aff_score = torch.nn.functional.cosine_similarity(sp_scores.unsqueeze(0), sp_scores.unsqueeze(1), dim=-1)
    #
    #         aff_color = torch.nn.functional.cosine_similarity(RGB_features.unsqueeze(0), RGB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(HSV_features.unsqueeze(0), HSV_features.unsqueeze(1), dim=-1)
    #
    #
    #         aff_mat = aff_sp_f + aff_color + aff_score
    #
    #         aff_mat_mask = aff_mat * (torch.tensor(adjacent_matrix, dtype=torch.float32) + torch.eye(adjacent_matrix.shape[0]))
    #
    #         trans_mat = aff_mat_mask / (torch.sum(aff_mat_mask, dim=0, keepdim=True)+1e-5)
    #
    #         plt.close()
    #         fig,axx = plt.subplots(2, 3)
    #
    #         sns.heatmap(aff_sp_f.numpy(),cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,0])
    #         axx[0,0].set_title("aff_sp_f")
    #
    #         sns.heatmap(aff_score.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,1])
    #         axx[0,1].set_title("aff_score")
    #
    #         # axx[0,2].axis("off")
    #         sns.heatmap(aff_color.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,2])
    #         axx[0,2].set_title("aff_color")
    #
    #         sns.heatmap(aff_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,0])
    #         axx[1,0].set_title("aff_mat")
    #
    #         sns.heatmap(aff_mat_mask.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,1])
    #         axx[1,1].set_title("aff_mat_mask")
    #
    #         sns.heatmap(trans_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,2])
    #         axx[1,2].set_title("trans_mat")
    #
    #         print(os.path.join(dir,str(img_id)+"_aff.jpg"))
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+"_aff.jpg"), dpi=300)
    #
    #         # 包括背景类 --> (21, sp_num)
    #         # RW_cams = torch.tensor(sp_scores_normalize.t(), dtype=torch.float32)
    #         RW_sp_cams =  torch.tensor(sp_cams.t(), dtype=torch.float32)
    #         RW_sp_normalize_cams = torch.tensor(sp_scores_normalize.t(), dtype=torch.float32)
    #
    #         trans_mat = torch.tensor(trans_mat, dtype=torch.float32)
    #
    #         cam_valid_peak_list, cam_peak_score = self.get_peak_list(class_response_maps,lables,peak_threshold)
    #         response_maps = [class_response_maps]
    #         valid_peak_list = [cam_valid_peak_list]
    #         scale = 0.7
    #
    #         iter_TIME = 4 + 1
    #
    #         for i in range(iter_TIME):
    #             if i == 0:
    #                 RW_sp_cams = RW_sp_cams
    #                 RW_sp_normalize_cams = RW_sp_normalize_cams
    #             else:
    #                 RW_sp_cams = torch.matmul(RW_sp_cams, trans_mat)
    #                 RW_sp_normalize_cams = torch.matmul(RW_sp_normalize_cams, trans_mat)
    #
    #             RW_sp_normalize_cams = torch.nn.functional.normalize(RW_sp_normalize_cams, p=1, dim=0)
    #
    #             RW_sp_cams_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_sp_cams.dtype)
    #             RW_sp_normalize_cams_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_sp_normalize_cams.dtype)
    #
    #             for cls_idx in lables:
    #                 for idx, temp_label in enumerate(range(low_label, high_label + 1)):
    #                     sp_mask = sp_label == temp_label
    #                     sp_mask_tensor = torch.tensor(sp_mask)
    #                     # exclusive bg class
    #                     RW_sp_cams_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_sp_cams[cls_idx,idx] # single batch input
    #                     RW_sp_normalize_cams_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_sp_normalize_cams[cls_idx + 1,idx] # single batch input
    #
    #             RW_class_response_maps = RW_sp_cams_class_response_maps * RW_sp_normalize_cams_class_response_maps
    #
    #             RW_class_response_maps = F.upsample(RW_class_response_maps,size=(cam_H,cam_W),mode='bilinear', align_corners=True)
    #
    #             RW_valid_peak_list, RW_peak_score = self.get_peak_list(RW_class_response_maps,lables,0,self._lzc_filter)
    #
    #             response_maps.append(RW_class_response_maps)
    #             valid_peak_list.append(RW_valid_peak_list)
    #
    #         RW_class_response_maps = RW_class_response_maps.numpy()
    #         RW_class_response_maps = RW_class_response_maps - np.min(RW_class_response_maps,(-2,-1),keepdims = True)
    #         RW_class_response_maps = RW_class_response_maps / np.max(RW_class_response_maps,(-2,-1),keepdims = True)
    #         RW_class_response_maps = torch.tensor(RW_class_response_maps)
    #
    #         print(RW_class_response_maps)
    #         mask_CAM = RW_class_response_maps * class_response_maps
    #         mask_CAM_valid_peak_list, mask_CAM_peak_score = self.get_peak_list(mask_CAM, lables, peak_threshold)
    #         response_maps.append(mask_CAM)
    #         valid_peak_list.append(mask_CAM_valid_peak_list)
    #
    #         cam_num = onehot_label.sum().item()
    #         plt.close()
    #         f, axarr = plt.subplots(len(response_maps), int(cam_num + 1))
    #         plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.1, hspace=0.1)
    #
    #         for idx in range(len(response_maps)):
    #             if idx == 0:
    #                 axarr[idx, 0].imshow(img)
    #             else:
    #                 axarr[idx, 0].imshow(mark_boundaries(img, sp_label))
    #
    #             axarr[idx, 0].axis("off")
    #
    #             for i,cls_idx in enumerate(lables):
    #                 CRM = response_maps[idx]
    #
    #                 cam = CRM[0, cls_idx].detach().cpu().numpy()
    #                 cam = cam - np.min(cam)
    #                 cam = cam / np.max(cam)
    #                 cam = np.uint8(255 * cam)
    #
    #                 heatmap = cv2.applyColorMap(cv2.resize(cam, (W, H)), cv2.COLORMAP_JET)
    #                 heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #                 result = heatmap * scale + (img) * (1 - scale)
    #                 result = np.uint8(result)
    #                 result = cv2.resize(result, (int(cam_W*self.sub_pixel_locating_factor),int(cam_H*self.sub_pixel_locating_factor)))
    #
    #                 axarr[idx, i+1].imshow(result)
    #                 axarr[idx, i+1].axis("off")
    #                 axarr[idx, i+1].set_title(voc_id_name_map[cls_idx.item()+1])
    #
    #                 VPL = valid_peak_list[idx]
    #
    #                 for peak in VPL:
    #                     if peak[1] == cls_idx:
    #                         axarr[idx, i+1].scatter(peak[3],peak[2], color='purple',marker='*',edgecolors='purple', s=17)
    #
    #                 axarr[idx, i+1].axis("off")
    #
    #         # plt.show()
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+".jpg"), dpi=300)
    #
    #         # if len(peak_score) > 0:
    #         #     valid_peak_list = torch.stack(valid_peak_list)
    #         #     peak_score = torch.tensor(peak_score)
    #         #     # classification confidence scores, class-aware and instance-aware visual cues
    #         #     return class_response_maps, valid_peak_list, peak_score
    #         # else:
    #         #     return None

    # def forward(self, input, lables,img,class_threshold=0, peak_threshold=10, retrieval_cfg=None,graph_op=None,resolution=7):
    #     assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
    #     with torch.no_grad():
    #         device = input.device
    #         _, _, H, W = input.shape  # batchsize == 1
    #
    #         # classification network forwarding
    #         class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
    #
    #
    #         _, cls_num, cam_H, cam_W = class_response_maps.shape
    #
    #         onehot_label = torch.tensor(self.label2onehot(lables,cls_num)).to(device)
    #         class_response_maps = class_response_maps * torch.tensor(onehot_label).view(1,cls_num,1,1)
    #
    #         org_H,org_W,_ = img.shape
    #         org_size_class_response_maps = F.upsample(class_response_maps, size=(org_H,org_W), mode='bilinear', align_corners=True)
    #
    #         # use p2 as feature
    #         feature = p2
    #
    #         _, _, f_H, f_W = feature.shape
    #
    #         # spatial_scale
    #         spatial_scale = f_H / H
    #         Align_op = RoIAlign((resolution, resolution), spatial_scale, 0)
    #
    #         # graph operate
    #         sp_label = graph_op["sp_label"]
    #         adjacent_matrix = graph_op["adjacent_matrix"]
    #
    #         sp_num = adjacent_matrix.shape[0]
    #
    #         for idx, temp_label in enumerate(range(sp_num)):
    #             sp_mask = sp_label == temp_label
    #
    #             sp_mask_tensor = torch.tensor(sp_mask).to(device)
    #             cam_scores = (sp_mask_tensor * org_size_class_response_maps).sum(-1).sum(-1) / sp_mask_tensor.sum() # --> (1, cls_num)
    #
    #             ind_xy = np.nonzero(sp_mask)
    #             xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
    #
    #             mask = sp_mask[ymin:ymax, xmin:xmax]
    #             mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
    #             mask = torch.tensor(mask).to(device)
    #
    #             box = torch.tensor([[0, xmin, ymin, xmax, ymax]],dtype=torch.float32).to(device)
    #             box_feature = Align_op(feature, box)  # (1,7,7)
    #
    #             sp_f = (mask * box_feature).view(-1, resolution, resolution)  # (1,dim,7,7) --> (dim,7,7)
    #             sp_f = sp_f.view(-1, resolution * resolution).sum(-1)  # (dim)
    #             sp_f = sp_f / mask.sum()
    #
    #             sp_f = sp_f[None, :]
    #
    #             if idx == 0:
    #                 sp_cams = cam_scores
    #                 sp_features = sp_f
    #             else:
    #                 sp_cams = torch.cat((sp_cams, cam_scores), dim=0)
    #                 sp_features = torch.cat((sp_features, sp_f), dim=0)
    #
    #
    #         # sp_cam --> (sp_num, cls_num)
    #         # sp_features --> (sp_num, dim)
    #         # rgb_features --> (sp_num, 3)
    #
    #         # make similarity
    #         aff_mat = torch.nn.functional.cosine_similarity(sp_features.unsqueeze(0), sp_features.unsqueeze(1), dim=-1)
    #
    #         aff_mat = aff_mat * (torch.tensor(adjacent_matrix,dtype=torch.float32)+torch.eye(adjacent_matrix.shape[0])).to(device)
    #         trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    #
    #         RW_cams = torch.matmul(torch.tensor(sp_cams.t(),dtype=torch.float32), torch.tensor(trans_mat,dtype=torch.float32)) # cls_num, sp_num
    #         RW_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_cams.dtype).to(device)
    #         RW_class_response_maps = RW_class_response_maps.cpu()
    #         RW_cams = RW_cams.cpu()
    #         class_response_maps = class_response_maps.cpu()
    #
    #         for cls_idx in lables:
    #             for idx, temp_label in enumerate(range(sp_num)):
    #                 sp_mask = sp_label == temp_label
    #                 sp_mask_tensor = torch.tensor(sp_mask)
    #                 RW_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_cams[cls_idx,temp_label] # single batch input
    #
    #         RW_class_response_maps = F.upsample(RW_class_response_maps,size=(cam_H,cam_W),mode='bilinear', align_corners=True)
    #
    #         # a = 0.5
    #         # smooth_class_response_maps = a * RW_class_response_maps+(1-a)*class_response_maps
    #
    #         # cam_valid_peak_list, cam_peak_score = self.get_peak_list(class_response_maps,lables,peak_threshold)
    #         RW_valid_peak_list, RW_peak_score = self.get_peak_list(RW_class_response_maps,lables,peak_threshold)
    #         # smooth_valid_peak_list, smooth_peak_score = self.get_peak_list(smooth_class_response_maps, lables, peak_threshold)
    #
    #         # if len(smooth_peak_score) > 0:
    #         #     valid_peak_list = torch.stack(smooth_valid_peak_list)
    #         #     peak_score = torch.tensor(smooth_peak_score)
    #         #     # classification confidence scores, class-aware and instance-aware visual cues
    #         #     return smooth_class_response_maps, valid_peak_list, peak_score
    #         if len(RW_peak_score) > 0:
    #             valid_peak_list = torch.stack(RW_valid_peak_list)
    #             peak_score = torch.tensor(RW_peak_score)
    #             # classification confidence scores, class-aware and instance-aware visual cues
    #             return RW_class_response_maps, valid_peak_list, peak_score
    #         else:
    #             return None

    def train(self, mode=True):
        super(PeakResponseMapping_aff, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super(PeakResponseMapping_aff, self).train(False)
        self._patch()
        self.inferencing = True
        return self

    def label2onehot(self, cls_label, cls_num):
        onehot_label = np.zeros(cls_num)
        for cls_idx in cls_label:
            onehot_label[cls_idx] = 1

        return onehot_label

    def get_peak_list(self, class_response_maps, lables, peak_threshold, peak_filter=None):
        if self.enable_peak_stimulation:
            # sub-pixel peak finding
            if self.sub_pixel_locating_factor > 1:
                class_response_maps = F.upsample(class_response_maps, scale_factor=self.sub_pixel_locating_factor,
                                                 mode='bilinear', align_corners=True)
            # aggregate responses from informative receptive fields estimated via class peak responses
            if peak_filter == None:
                peak_list = peak_stimulation_aff(class_response_maps, win_size=self.win_size,
                                                 peak_filter=self.peak_filter)
            else:
                peak_list = peak_stimulation_aff(class_response_maps, win_size=self.win_size, peak_filter=peak_filter)
        else:
            # aggregate responses from all receptive fields
            peak_list = F.adaptive_avg_pool2d(class_response_maps, 1).squeeze(2).squeeze(2)

        # extract instance-aware visual cues, i.e., peak response maps
        assert class_response_maps.size(
            0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'
        if peak_list is None:
            if peak_filter == None:
                peak_list = peak_stimulation_aff(class_response_maps, return_aggregation=False, win_size=self.win_size,
                                                 peak_filter=self.peak_filter)
            else:
                peak_list = peak_stimulation_aff(class_response_maps, return_aggregation=False, win_size=self.win_size,
                                                 peak_filter=peak_filter)
            # peak_list = peak_stimulation_aff(class_response_maps, return_aggregation=False, win_size=self.win_size,peak_filter=self.peak_filter)

        valid_peak_list = []
        peak_score = []
        for cls_idx in lables:
            peak_list_cls = peak_list[peak_list[:, 1] == cls_idx]
            # print(peak_list_cls)
            exist = 0
            for idx in range(peak_list_cls.size(0)):
                peak_val = class_response_maps[
                    peak_list_cls[idx, 0], peak_list_cls[idx, 1], peak_list_cls[idx, 2], peak_list_cls[idx, 3]]
                # print(peak_val)
                if peak_val > peak_threshold:
                    exist = 1
                    # starting from the peak
                    valid_peak_list.append(peak_list_cls[idx, :])
                    peak_score.append(peak_val)
            # if exist == 0:
            #     peak_val_list = [class_response_maps[0, cls_idx, peak_list_cls[idx, 2], peak_list_cls[idx, 3]] for
            #                      idx in range(peak_list_cls.size(0))]
            #     peak_val_list = torch.tensor(peak_val_list)
            #     peak_val = class_response_maps[peak_list_cls[peak_val_list.argmax(), 0], \
            #                                    peak_list_cls[peak_val_list.argmax(), 1], \
            #                                    peak_list_cls[peak_val_list.argmax(), 2], \
            #                                    peak_list_cls[peak_val_list.argmax(), 3]]
            #     valid_peak_list.append(peak_list_cls[peak_val_list.argmax(), :])
            #     peak_score.append(peak_val)

        return valid_peak_list, peak_score


def peak_response_mapping_aff_vis(
        backbone: nn.Module = fc_resnet50_aff(),
        enable_peak_stimulation: bool = True,
        enable_peak_backprop: bool = True,
        win_size: int = 3,
        sub_pixel_locating_factor: int = 8,
        filter_type: Union[str, int, float] = 'median') -> nn.Module:
    """Peak Response Mapping.
    """

    model = PeakResponseMapping_aff_visual(
        backbone,
        enable_peak_stimulation=enable_peak_stimulation,
        enable_peak_backprop=enable_peak_backprop,
        win_size=win_size,
        sub_pixel_locating_factor=sub_pixel_locating_factor,
        filter_type=filter_type)
    return model


# only for inferencing visual
class PeakResponseMapping_aff_visual(nn.Sequential):

    def __init__(self, *args, **kargs):
        super(PeakResponseMapping_aff_visual, self).__init__(*args)

        self.inferencing = True
        # use global average pooling to aggregate responses if peak stimulation is disabled
        self.enable_peak_stimulation = kargs.get('enable_peak_stimulation', True)
        # return only the class response maps in inference mode if peak backpropagation is disabled
        self.enable_peak_backprop = kargs.get('enable_peak_backprop', True)
        # window size for peak finding
        self.win_size = kargs.get('win_size', 3)
        # sub-pixel peak finding
        self.sub_pixel_locating_factor = kargs.get('sub_pixel_locating_factor', 1)
        # peak filtering
        self.filter_type = kargs.get('filter_type', 'median')
        if self.filter_type == 'median':
            self.peak_filter = self._median_filter
        elif self.filter_type == 'mean':
            self.peak_filter = self._mean_filter
        elif self.filter_type == 'max':
            self.peak_filter = self._max_filter
        elif isinstance(self.filter_type, (int, float)):
            self.peak_filter = lambda x: self.filter_type
        else:
            self.peak_filter = None

    @staticmethod
    def _median_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _mean_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _max_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _lzc_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return 0.5 * threshold.contiguous().view(batch_size, num_channels, 1, 1)

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward

    def instance_nms(self, instance_list, threshold=0.3, merge_peak_response=True):
        selected_instances = []
        while len(instance_list) > 0:
            instance = instance_list.pop(0)
            selected_instances.append(instance)
            src_mask = instance[2].astype(bool)
            src_peak_response = instance[3]

            def iou_filter(x):
                dst_mask = x[2].astype(bool)
                # IoU
                intersection = np.logical_and(src_mask, dst_mask).sum()
                union = np.logical_or(src_mask, dst_mask).sum()
                iou = intersection / (union + 1e-10)
                if iou < threshold:
                    return x
                else:
                    if merge_peak_response:
                        nonlocal src_peak_response
                        src_peak_response += x[3]
                    return None

            instance_list = list(filter(iou_filter, instance_list))
        return selected_instances

    def instance_seg(self, class_response_maps, peak_list, peak_response_maps, retrieval_cfg):
        # cast tensors to numpy array
        class_response_maps = class_response_maps.squeeze().cpu().numpy()
        peak_list = peak_list.cpu().numpy()
        peak_response_maps = peak_response_maps.cpu().numpy()

        img_height, img_width = peak_response_maps.shape[1], peak_response_maps.shape[2]

        # image size
        img_area = img_height * img_width

        # segment proposals off-the-shelf
        proposals = retrieval_cfg['proposals']

        # proposal contour width
        contour_width = retrieval_cfg.get('contour_width', 5)

        # limit range of proposal size
        proposal_size_limit = retrieval_cfg.get('proposal_size_limit', (0.00002, 0.85))

        # selected number of proposals
        proposal_count = retrieval_cfg.get('proposal_count', 100)

        # nms threshold
        nms_threshold = retrieval_cfg.get('nms_threshold', 0.3)

        # merge peak response during nms
        merge_peak_response = retrieval_cfg.get('merge_peak_response', True)

        # metric free parameters
        param = retrieval_cfg.get('param', None)

        # process each peak
        instance_list = []
        for i in range(len(peak_response_maps)):
            class_idx = peak_list[i, 1]

            # extract hyper-params
            if isinstance(param, tuple):
                # shared param
                bg_threshold_factor, penalty_factor, balance_factor = param
            elif isinstance(param, list):
                # independent params between classes
                bg_threshold_factor, penalty_factor, balance_factor = param[class_idx]
            else:
                raise TypeError('Invalid hyper-params "%s".' % param)

            class_response = imresize(class_response_maps[class_idx], (img_height, img_width), interp='bicubic')
            bg_response = (class_response < bg_threshold_factor * class_response.mean()).astype(np.float32)
            peak_response_map = peak_response_maps[i]

            # select proposal
            max_val = -np.inf
            instance_mask = None

            for j in range(min(proposal_count, len(proposals))):
                raw_mask = imresize(proposals[j].astype(int), peak_response_map.shape, interp='nearest')
                # get contour of the proposal
                contour_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_GRADIENT,
                                                np.ones((contour_width, contour_width), np.uint8)).astype(bool)
                mask = raw_mask.astype(bool)
                # metric
                mask_area = mask.sum()
                if (mask_area >= proposal_size_limit[1] * img_area) or \
                        (mask_area < proposal_size_limit[0] * img_area):
                    continue
                else:
                    val = balance_factor * peak_response_map[mask].sum() + \
                          peak_response_map[contour_mask].sum() - \
                          penalty_factor * bg_response[mask].sum()
                    if val > max_val:
                        max_val = val
                        instance_mask = mask

            if instance_mask is not None:
                instance_list.append((max_val, class_idx, instance_mask, peak_response_map))

        instance_list = sorted(instance_list, key=lambda x: x[0], reverse=True)
        if nms_threshold is not None:
            instance_list = self.instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), nms_threshold,
                                              merge_peak_response)
        return [dict(score=v[0], category=v[1], mask=v[2], prm=v[3]) for v in instance_list]

    # use proposal score to replace cam score
    # draw
    def forward(self, input, lables, img, class_threshold=0, peak_threshold=10, retrieval_cfg=None, graph_op=None,
                resolution=7):
        assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
        dir = "./voc_visual"
        if not os.path.exists(dir):
            os.makedirs(dir)
        with torch.no_grad():

            # _, _, H, W = input.shape  # batchsize == 1
            H, W, _ = np.array(img).shape

            LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            HSV_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # classification network forwarding
            class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff_visual, self).forward(input)
            class_response_maps = class_response_maps.detach()
            org_size_class_response_maps = F.upsample(class_response_maps, size=(H, W), mode='bilinear',
                                                      align_corners=True)

            _, cls_num, cam_H, cam_W = class_response_maps.shape

            onehot_label = self.label2onehot(lables, cls_num)
            class_response_maps = class_response_maps * torch.tensor(onehot_label).view(1, cls_num, 1, 1)

            _, _, f_H_p1, f_W_p1 = p1.shape
            _, _, f_H_p2, f_W_p2 = p2.shape
            _, _, f_H_p3, f_W_p3 = p3.shape
            _, _, f_H_p4, f_W_p4 = p4.shape

            Align_op_p1 = RoIAlign((resolution, resolution), f_H_p1 / H, 0)
            Align_op_p2 = RoIAlign((resolution, resolution), f_H_p2 / H, 0)
            Align_op_p3 = RoIAlign((resolution, resolution), f_H_p3 / H, 0)
            Align_op_p4 = RoIAlign((resolution, resolution), f_H_p4 / H, 0)

            # graph operate
            sp_label = graph_op["sp_label"]
            adjacent_matrix = graph_op["adjacent_matrix"]
            img_id = graph_op["img_id"]
            belong_mat = graph_op["belong_mat"]
            proposal_score = graph_op["proposal_score"]
            cocoGT = graph_op["cocoGT"]
            proposal_sum = graph_op["proposal_sum"]
            proposal = graph_op["proposal"]
            print(img_id)

            # add bg_score
            # bg_score = 1 - proposal_score.sum(-1)
            # proposal_score = np.concatenate((bg_score[:,None],proposal_score),axis=1)
            #
            # sp_scores = np.zeros((adjacent_matrix.shape[0],proposal_score.shape[-1])) # sp_num, cls_num + 1
            # for row_idx in range(belong_mat.shape[0]):
            #     score = proposal_score[row_idx]
            #     belong_row = belong_mat[row_idx] != 0
            #
            #     sp_scores[belong_row] += score
            #
            # sp_scores = torch.tensor(sp_scores)
            # sp_scores_normalize = torch.nn.functional.normalize(sp_scores,p=1,dim=-1)

            # add bg_score
            # bg_score = 1 - proposal_score.sum(-1)
            # proposal_score = np.concatenate((bg_score[:, None], proposal_score), axis=1)

            # max_idx = np.argmax(proposal_score, axis=-1)
            # flag = np.zeros_like(max_idx)
            # for label in lables:
            #     flag += (label + 1).item() == max_idx
            #
            # bg_score_const = np.zeros(proposal_score.shape[-1])
            # bg_score_const[0] = 1
            # proposal_score[flag == 0] = bg_score_const
            #
            # sp_scores = np.zeros((adjacent_matrix.shape[0], proposal_score.shape[-1]))  # sp_num, cls_num + 1
            # for row_idx in range(belong_mat.shape[0]):
            #     score = proposal_score[row_idx]
            #     belong_row = belong_mat[row_idx] != 0
            #
            #     sp_scores[belong_row] += score
            #
            # sp_scores = torch.tensor(sp_scores)
            # sp_scores_normalize = torch.nn.functional.normalize(sp_scores, p=1, dim=-1)

            low_label = np.min(sp_label)
            high_label = np.max(sp_label)

            # for idx, temp_label in tqdm(enumerate(range(low_label, high_label + 1)),total=adjacent_matrix.shape[0]):
            #     sp_mask = sp_label == temp_label
            #     sp_mask_tensor = torch.tensor(sp_mask)
            #
            #     if idx == 0:
            #         sp_proposal = sp_mask[None,:,:]
            #     else:
            #         sp_proposal = np.concatenate((sp_proposal, sp_mask[None,:,:]),axis=0)
            #     cam_scores = (sp_mask_tensor * org_size_class_response_maps).sum((-2,-1)) / sp_mask_tensor.sum() # --> (1, cls_num)
            #
            #
            #     ind_xy = np.nonzero(sp_mask)
            #     xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
            #
            #     mask = sp_mask[ymin:ymax, xmin:xmax]
            #     mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
            #     mask = torch.tensor(mask)
            #
            #     box = torch.tensor([[0, xmin, ymin, xmax, ymax]],dtype=torch.float32)
            #     box_feature_p1 = Align_op_p1(p1, box)  # (1,7,7)
            #     box_feature_p2 = Align_op_p2(p2, box)  # (1,7,7)
            #     box_feature_p3 = Align_op_p3(p3, box)  # (1,7,7)
            #     box_feature_p4 = Align_op_p4(p4, box)  # (1,7,7)
            #
            #     sp_feature_p1 = get_sp_f(box_feature_p1,mask,resolution)
            #     sp_feature_p2 = get_sp_f(box_feature_p2,mask,resolution)
            #     sp_feature_p3 = get_sp_f(box_feature_p3,mask,resolution)
            #     sp_feature_p4 = get_sp_f(box_feature_p4,mask,resolution)
            #
            #     if idx == 0:
            #         sp_cams = cam_scores
            #         sp_features_p1 = sp_feature_p1
            #         sp_features_p2 = sp_feature_p2
            #         sp_features_p3 = sp_feature_p3
            #         sp_features_p4 = sp_feature_p4
            #
            #         RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
            #         LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
            #         HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
            #         for color_idx in range(1,3):
            #             temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
            #             temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
            #             temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
            #
            #             RGB_feature = np.concatenate((RGB_feature,temp_feature_rgb),axis=0)
            #             LAB_feature = np.concatenate((LAB_feature,temp_feature_lab),axis=0)
            #             HSV_feature = np.concatenate((HSV_feature,temp_feature_hsv),axis=0)
            #
            #         RGB_features = RGB_feature.transpose()
            #         LAB_features = LAB_feature.transpose()
            #         HSV_features = HSV_feature.transpose()
            #
            #     else:
            #         sp_cams = torch.cat((sp_cams, cam_scores), dim=0)
            #         sp_features_p1 = torch.cat((sp_features_p1, sp_feature_p1),dim=0)
            #         sp_features_p2 = torch.cat((sp_features_p2, sp_feature_p2),dim=0)
            #         sp_features_p3 = torch.cat((sp_features_p3, sp_feature_p3),dim=0)
            #         sp_features_p4 = torch.cat((sp_features_p4, sp_feature_p4),dim=0)
            #
            #         RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
            #         LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
            #         HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
            #         for color_idx in range(1, 3):
            #             temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
            #             temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
            #             temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
            #
            #             RGB_feature = np.concatenate((RGB_feature, temp_feature_rgb), axis=0)
            #             LAB_feature = np.concatenate((LAB_feature, temp_feature_lab), axis=0)
            #             HSV_feature = np.concatenate((HSV_feature, temp_feature_hsv), axis=0)
            #
            #         RGB_features = np.concatenate((RGB_features,RGB_feature.transpose()),axis=0)
            #         LAB_features = np.concatenate((LAB_features,LAB_feature.transpose()),axis=0)
            #         HSV_features = np.concatenate((HSV_features,HSV_feature.transpose()),axis=0)

            # RGB_features = torch.tensor(RGB_features)
            # LAB_features = torch.tensor(LAB_features)
            # HSV_features = torch.tensor(HSV_features)
            #
            # # make similarity
            # aff_sp_f = torch.nn.functional.cosine_similarity(sp_features_p1.unsqueeze(0), sp_features_p1.unsqueeze(1), dim=-1) * \
            #            torch.nn.functional.cosine_similarity(sp_features_p2.unsqueeze(0), sp_features_p2.unsqueeze(1),dim=-1) * \
            #            torch.nn.functional.cosine_similarity(sp_features_p3.unsqueeze(0), sp_features_p3.unsqueeze(1),dim=-1) * \
            #            torch.nn.functional.cosine_similarity(sp_features_p4.unsqueeze(0), sp_features_p4.unsqueeze(1),dim=-1)
            # # aff_sp_f = torch.nn.functional.cosine_similarity(sp_features_p4.unsqueeze(0), sp_features_p4.unsqueeze(1),dim=-1)
            #
            # aff_score = torch.nn.functional.cosine_similarity(sp_scores.unsqueeze(0), sp_scores.unsqueeze(1), dim=-1)
            #
            # aff_color = torch.nn.functional.cosine_similarity(RGB_features.unsqueeze(0), RGB_features.unsqueeze(1), dim=-1) * \
            #             torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1), dim=-1) * \
            #             torch.nn.functional.cosine_similarity(HSV_features.unsqueeze(0), HSV_features.unsqueeze(1), dim=-1)
            # # aff_color = torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1), dim=-1)
            #
            # aff_mat = aff_sp_f + aff_color + aff_score
            # # aff_mat = aff_sp_f * aff_color
            # # aff_mat = aff_sp_f * aff_score * aff_rgb
            #
            # aff_mat_mask = aff_mat * (torch.tensor(adjacent_matrix, dtype=torch.float32) + torch.eye(adjacent_matrix.shape[0]))
            #
            # trans_mat = aff_mat_mask / (torch.sum(aff_mat_mask, dim=0, keepdim=True)+1e-5)
            # # D = np.diag(aff_mat_mask.sum(-1))
            # # D_1 = torch.tensor(np.sqrt(np.linalg.pinv(D)))
            # # trans_mat = D_1 @ aff_mat_mask @ D_1
            #
            # plt.close()
            # fig,axx = plt.subplots(2, 3)
            #
            # sns.heatmap(aff_sp_f.numpy(),cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,0])
            # axx[0,0].set_title("aff_sp_f")
            #
            # sns.heatmap(aff_score.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,1])
            # axx[0,1].set_title("aff_score")
            #
            # # axx[0,2].axis("off")
            # sns.heatmap(aff_color.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,2])
            # axx[0,2].set_title("aff_color")
            #
            # sns.heatmap(aff_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,0])
            # axx[1,0].set_title("aff_mat")
            #
            # sns.heatmap(aff_mat_mask.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,1])
            # axx[1,1].set_title("aff_mat_mask")
            #
            # sns.heatmap(trans_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,2])
            # axx[1,2].set_title("trans_mat")
            #
            # print(os.path.join(dir,str(img_id)+"_aff.jpg"))
            #
            # plt.savefig(os.path.join(dir,str(img_id)+"_aff.jpg"), dpi=300)

            # # 包括背景类 --> (21, sp_num)
            # # RW_cams = torch.tensor(sp_cams.t(), dtype=torch.float32)
            # RW_cams = torch.tensor(sp_scores_normalize.t(), dtype=torch.float32)
            # trans_mat = torch.tensor(trans_mat, dtype=torch.float32)

            cam_valid_peak_list, cam_peak_score = self.get_peak_list(class_response_maps, lables, peak_threshold)
            response_maps = [class_response_maps]
            valid_peak_list = [cam_valid_peak_list]
            scale = 0.7

            # RW_cams = torch.matmul(torch.tensor(trans_mat.t(),dtype=torch.float32),torch.tensor(sp_cams,dtype=torch.float32))
            # RW_cams = RW_cams.t()

            iter_TIME = 4 + 1

            # for i in range(1):
            #     if i == 0:
            #         RW_cams = RW_cams
            #     else:
            #         RW_cams =  torch.matmul(RW_cams, trans_mat)
            #     RW_cams = torch.nn.functional.normalize(RW_cams, p=1, dim=0)
            #
            #     RW_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_cams.dtype)
            #
            #     for cls_idx in lables:
            #         for idx, temp_label in enumerate(range(low_label, high_label + 1)):
            #             sp_mask = sp_label == temp_label
            #             sp_mask_tensor = torch.tensor(sp_mask)
            #             # exclusive bg class
            #             RW_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_cams[cls_idx+1,idx] # single batch input
            #
            #     RW_class_response_maps = F.upsample(RW_class_response_maps,size=(cam_H,cam_W),mode='bilinear', align_corners=True)
            #
            #     RW_valid_peak_list, RW_peak_score = self.get_peak_list(RW_class_response_maps,lables,0,self._lzc_filter)
            #
            #     response_maps.append(RW_class_response_maps)
            #     valid_peak_list.append(RW_valid_peak_list)
            #
            # RW_class_response_maps = RW_class_response_maps.numpy()
            # RW_class_response_maps = RW_class_response_maps - np.min(RW_class_response_maps,(-2,-1),keepdims = True)
            # RW_class_response_maps = RW_class_response_maps / np.max(RW_class_response_maps,(-2,-1),keepdims = True)
            # RW_class_response_maps = torch.tensor(RW_class_response_maps)

            # # print(RW_class_response_maps)
            # mask_CAM = RW_class_response_maps * class_response_maps
            # mask_CAM_valid_peak_list, mask_CAM_peak_score = self.get_peak_list(mask_CAM, lables, peak_threshold)
            # response_maps.append(mask_CAM)
            # valid_peak_list.append(mask_CAM_valid_peak_list)

            cam_num = onehot_label.sum().item()
            plt.close()
            # f, axarr = plt.subplots(len(response_maps), int(cam_num + 1))
            # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.1, hspace=0.1)

            # print(proposal.shape)
            # for proposal_idx,propo in enumerate(proposal):
            #     # heatmap = cv2.applyColorMap(np.uint8(255. * propo), cv2.COLORMAP_JET)
            #     # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            #     # result = heatmap * scale + (img) * (1 - scale)
            #     # plt.close()
            #     # plt.imshow(np.uint8(result))
            #     # plt.axis("off")
            #     # if not os.path.exists(os.path.join(dir,str(img_id))):
            #     #     os.makedirs(os.path.join(dir,str(img_id)))
            #     # plt.savefig(os.path.join(dir,str(img_id),"proposal{}".format(proposal_idx) + ".jpg"), dpi=300, bbox_inches='tight',pad_inches=0)
            #
            #     plt.close()
            #     plt.imshow(np.uint8(255 * np.concatenate((propo[...,None],propo[...,None],propo[...,None]),axis=-1)))
            #     plt.axis("off")
            #     if not os.path.exists(os.path.join(dir, str(img_id))):
            #         os.makedirs(os.path.join(dir, str(img_id)))
            #     plt.savefig(os.path.join(dir, str(img_id), "proposal{}".format(proposal_idx) + ".jpg"), dpi=300,
            #                 bbox_inches='tight', pad_inches=0)

            HEATMAP = []
            for idx in range(1):
                if idx == 0:
                    img_col = cv2.resize(img, (
                    int(cam_W * self.sub_pixel_locating_factor), int(cam_H * self.sub_pixel_locating_factor)))

                    plt.close()
                    plt.imshow(np.uint8(img))
                    plt.axis("off")

                    plt.savefig(os.path.join(dir, str(img_id) + "org" + ".jpg"), dpi=300, bbox_inches='tight',
                                pad_inches=0)

                elif idx == 1:
                    sp_label = cv2.resize(np.uint8(sp_label), (
                    int(cam_W * self.sub_pixel_locating_factor), int(cam_H * self.sub_pixel_locating_factor)),
                                          cv2.INTER_NEAREST)
                    res = mark_boundaries(img_col, sp_label) * 255
                    img_col = np.concatenate((img_col, res), axis=1)

                elif idx == 2:
                    polygons = []
                    box_list = []
                    label_list = []
                    gt_ann_ids = cocoGT.getAnnIds(imgIds=[img_id])
                    gt_anns = cocoGT.loadAnns(gt_ann_ids)
                    print(gt_anns)
                    for ann in gt_anns:
                        if 'segmentation' in ann:
                            # print("here")
                            # print(ann['segmentation'])
                            rle = maskUtils.frPyObjects(ann['segmentation'], img.shape[0], img.shape[1])
                            mask = maskUtils.decode(rle).transpose(2, 0, 1)
                            # print(mask)
                            # print(mask.shape)

                            polygons.append(mask)
                        label_list.append(ann['category_id'] - 1)
                        box_list.append(np.array(ann['bbox']))

                    polygons = np.concatenate(polygons, axis=0)
                    # print(polygons.shape)
                    label_list = np.array(label_list)
                    box_list = np.concatenate(box_list, axis=0).reshape((-1, 4))
                    # gt_img = imshow_det_bboxes(img,box_list,labels=label_list,segms=polygons,show=False)
                    gt_img = imshow_det_bboxes(img, labels=label_list, segms=polygons, show=False)

                    img_col = np.concatenate((img_col, cv2.resize(gt_img, (
                    int(cam_W * self.sub_pixel_locating_factor), int(cam_H * self.sub_pixel_locating_factor)))), axis=1)

                elif idx == 3:
                    print(proposal_sum.shape)
                    print(proposal_sum)
                    proposal_sum = np.uint8(255 * proposal_sum / proposal_sum.max())
                    heatmap = cv2.applyColorMap(proposal_sum, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    res = heatmap * scale + (img) * (1 - scale)

                    plt.close()
                    plt.imshow(np.uint8(res))
                    plt.axis("off")
                    plt.savefig(os.path.join(dir, str(img_id) + "proposal_sum" + ".jpg"), dpi=300, bbox_inches='tight',
                                pad_inches=0)

                    res = cv2.resize(res, (
                    int(cam_W * self.sub_pixel_locating_factor), int(cam_H * self.sub_pixel_locating_factor)))
                    img_col = np.concatenate((img_col, res), axis=1)

                else:
                    img_col = np.concatenate((img_col, 255 * np.ones(
                        (int(cam_W * self.sub_pixel_locating_factor), int(cam_H * self.sub_pixel_locating_factor), 3))),
                                             axis=1)

                for i, cls_idx in enumerate(lables):
                    CRM = response_maps[idx]

                    cam = CRM[0, cls_idx].detach().cpu().numpy()
                    cam = cam - np.min(cam)
                    cam = cam / np.max(cam)
                    cam = np.uint8(255 * cam)

                    heatmap = cv2.applyColorMap(cv2.resize(cam, (W, H)), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    result = heatmap * scale + (img) * (1 - scale)
                    result = np.uint8(result)
                    result = cv2.resize(result, (
                    int(cam_W * self.sub_pixel_locating_factor), int(cam_H * self.sub_pixel_locating_factor)))

                    VPL = valid_peak_list[idx]

                    for peak in VPL:
                        if peak[1] == cls_idx:
                            result = cv2.circle(result, center=(int(peak[3]), int(peak[2])), radius=2,
                                                color=(0, 0, 255), thickness=-1)

                    if idx == 0:
                        heat = result

                        plt.close()
                        plt.imshow(np.uint8(heat))
                        plt.axis("off")

                        plt.savefig(os.path.join(dir, str(img_id), str(img_id) + "heat" + ".jpg"), dpi=300,
                                    bbox_inches='tight', pad_inches=0)

                        print(len(VPL))
                        for idk, peak in enumerate(VPL):
                            if peak[1] == cls_idx:
                                result = cv2.circle(result, center=(int(peak[3]), int(peak[2])), radius=2,
                                                    color=(0, 0, 255), thickness=-1)
                                # mask_proposals: 200,375,500
                                x = int(peak[2] * proposal.shape[1] / 112)
                                y = int(peak[3] * proposal.shape[2] / 112)

                                avgmask = proposal[proposal[:, x, y] > 0, :, :].mean(0) > 0.7

                                heatmap = cv2.applyColorMap(np.uint8(avgmask * 1. * 255), cv2.COLORMAP_JET)
                                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                                res = heatmap * scale + (img) * (1 - scale)

                                plt.close()
                                plt.imshow(np.uint8(res))
                                plt.axis("off")
                                plt.savefig(os.path.join(dir, str(img_id), "avg_mask{}".format(idk) + ".jpg"), dpi=300,
                                            bbox_inches='tight', pad_inches=0)

                    if i == 0:
                        heat = result
                        plt.close()
                        result = cv2.resize(result, (
                        int(cam_W * self.sub_pixel_locating_factor), int(cam_H * self.sub_pixel_locating_factor)))

                        plt.imshow(np.uint8(result))
                        plt.axis("off")
                        plt.savefig(os.path.join(dir, str(img_id), "heat_{}".format(idk) + ".jpg"), dpi=300,
                                    bbox_inches='tight', pad_inches=0)

                    else:
                        heat = np.concatenate((heat, result), axis=0)

                HEATMAP.append(heat)

            HEATMAP = np.concatenate(HEATMAP, axis=1)
            result = np.concatenate((img_col, HEATMAP), axis=0)
            result = np.concatenate((255 * np.ones((result.shape[0], 80, 3)), result), axis=1)

            for i, cls_idx in enumerate(lables):
                cv2.putText(result, VOC_CLASSES[cls_idx], (5, (i) * 120 + 180), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (255, 0, 0), 1)

            plt.imshow(np.uint8(result))
            plt.axis("off")
            plt.savefig(os.path.join(dir, str(img_id), str(img_id) + ".jpg"), dpi=300, bbox_inches='tight')

            from pycocotools import mask as COCOMask
            import json

            def coco_encode(mask):
                encoding = COCOMask.encode(np.asfortranarray(mask))
                encoding['counts'] = encoding['counts'].decode('utf-8')
                return encoding

            # proposal_list = [31,119,146,13,104,]
            # proposal_list = [23,5,156,18]
            # proposal_list = [23, 97] # person
            proposal_list = [75, 169]

            json_results = []
            for proposal_idx in proposal_list:
                data = dict()
                data['image_id'] = int(img_id)
                data['segmentation'] = coco_encode(proposal[proposal_idx].astype(np.uint8))
                data['score'] = float(1)
                data['category_id'] = 1
                json_results.append(data)

            with open(os.path.join(dir, str(img_id), str(img_id) + ".json"), 'w') as f:
                f.write(json.dumps(json_results))

            # if len(peak_score) > 0:
            #     valid_peak_list = torch.stack(valid_peak_list)
            #     peak_score = torch.tensor(peak_score)
            #     # classification confidence scores, class-aware and instance-aware visual cues
            #     return class_response_maps, valid_peak_list, peak_score
            # else:
            #     return None

    def train(self, mode=True):
        super(PeakResponseMapping_aff_visual, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super(PeakResponseMapping_aff_visual, self).train(False)
        self._patch()
        self.inferencing = True
        return self

    def label2onehot(self, cls_label, cls_num):
        onehot_label = np.zeros(cls_num)
        for cls_idx in cls_label:
            onehot_label[cls_idx] = 1

        return onehot_label

    def get_peak_list(self, class_response_maps, lables, peak_threshold, peak_filter=None):
        if self.enable_peak_stimulation:
            # sub-pixel peak finding
            if self.sub_pixel_locating_factor > 1:
                class_response_maps = F.upsample(class_response_maps, scale_factor=self.sub_pixel_locating_factor,
                                                 mode='bilinear', align_corners=True)
            # aggregate responses from informative receptive fields estimated via class peak responses
            if peak_filter == None:
                peak_list = peak_stimulation_aff(class_response_maps, win_size=self.win_size,
                                                 peak_filter=self.peak_filter)
            else:
                peak_list = peak_stimulation_aff(class_response_maps, win_size=self.win_size, peak_filter=peak_filter)
        else:
            # aggregate responses from all receptive fields
            peak_list = F.adaptive_avg_pool2d(class_response_maps, 1).squeeze(2).squeeze(2)

        # extract instance-aware visual cues, i.e., peak response maps
        assert class_response_maps.size(
            0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'
        if peak_list is None:
            if peak_filter == None:
                peak_list = peak_stimulation_aff(class_response_maps, return_aggregation=False, win_size=self.win_size,
                                                 peak_filter=self.peak_filter)
            else:
                peak_list = peak_stimulation_aff(class_response_maps, return_aggregation=False, win_size=self.win_size,
                                                 peak_filter=peak_filter)
            # peak_list = peak_stimulation_aff(class_response_maps, return_aggregation=False, win_size=self.win_size,peak_filter=self.peak_filter)

        valid_peak_list = []
        peak_score = []
        for cls_idx in lables:
            peak_list_cls = peak_list[peak_list[:, 1] == cls_idx]
            # print(peak_list_cls)
            exist = 0
            for idx in range(peak_list_cls.size(0)):
                peak_val = class_response_maps[
                    peak_list_cls[idx, 0], peak_list_cls[idx, 1], peak_list_cls[idx, 2], peak_list_cls[idx, 3]]
                # print(peak_val)
                if peak_val > peak_threshold:
                    exist = 1
                    # starting from the peak
                    valid_peak_list.append(peak_list_cls[idx, :])
                    peak_score.append(peak_val)
            # if exist == 0:
            #     peak_val_list = [class_response_maps[0, cls_idx, peak_list_cls[idx, 2], peak_list_cls[idx, 3]] for
            #                      idx in range(peak_list_cls.size(0))]
            #     peak_val_list = torch.tensor(peak_val_list)
            #     peak_val = class_response_maps[peak_list_cls[peak_val_list.argmax(), 0], \
            #                                    peak_list_cls[peak_val_list.argmax(), 1], \
            #                                    peak_list_cls[peak_val_list.argmax(), 2], \
            #                                    peak_list_cls[peak_val_list.argmax(), 3]]
            #     valid_peak_list.append(peak_list_cls[peak_val_list.argmax(), :])
            #     peak_score.append(peak_val)

        return valid_peak_list, peak_score


# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from typing import Union, Optional, List, Tuple
from tqdm import tqdm

# %%
from types import MethodType
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.prm.prm_modules import pr_conv2d, peak_stimulation, peak_stimulation_aff
from mmcv.ops import RoIAlign
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
import os
from PIL import Image
from pycocotools import mask as maskUtils

VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')


def normalize_cam(class_response_maps_normalize):
    class_response_maps_normalize = (class_response_maps_normalize - class_response_maps_normalize.min((-2, -1))[
        ..., None, None]) / \
                                    (1e-5 + class_response_maps_normalize.max((-2, -1))[..., None, None] -
                                     class_response_maps_normalize.min((-2, -1))[..., None, None])

    print(class_response_maps_normalize)
    return class_response_maps_normalize


def get_sp_f(box_feature, mask, resolution):
    sp_f = (mask * box_feature).view(-1, resolution, resolution)  # (1,dim,7,7) --> (dim,7,7)
    sp_f = sp_f.sum((-2, -1))  # (dim)
    sp_f = sp_f / (mask.sum() + 1e-5)

    return sp_f[None, :]


class PeakResponseMapping(nn.Sequential):

    def __init__(self, *args, **kargs):
        super(PeakResponseMapping, self).__init__(*args)

        self.inferencing = False
        # use global average pooling to aggregate responses if peak stimulation is disabled
        self.enable_peak_stimulation = kargs.get('enable_peak_stimulation', True)
        # return only the class response maps in inference mode if peak backpropagation is disabled
        self.enable_peak_backprop = kargs.get('enable_peak_backprop', True)
        # window size for peak finding
        self.win_size = kargs.get('win_size', 3)
        # sub-pixel peak finding
        self.sub_pixel_locating_factor = kargs.get('sub_pixel_locating_factor', 1)
        # peak filtering
        self.filter_type = kargs.get('filter_type', 'median')
        if self.filter_type == 'median':
            self.peak_filter = self._median_filter
        elif self.filter_type == 'mean':
            self.peak_filter = self._mean_filter
        elif self.filter_type == 'max':
            self.peak_filter = self._max_filter
        elif isinstance(self.filter_type, (int, float)):
            self.peak_filter = lambda x: self.filter_type
        else:
            self.peak_filter = None

    @staticmethod
    def _median_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _mean_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _max_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward

    def instance_nms(self, instance_list, threshold=0.3, merge_peak_response=True):
        selected_instances = []
        while len(instance_list) > 0:
            instance = instance_list.pop(0)
            selected_instances.append(instance)
            src_mask = instance[2].astype(bool)
            src_peak_response = instance[3]

            def iou_filter(x):
                dst_mask = x[2].astype(bool)
                # IoU
                intersection = np.logical_and(src_mask, dst_mask).sum()
                union = np.logical_or(src_mask, dst_mask).sum()
                iou = intersection / (union + 1e-10)
                if iou < threshold:
                    return x
                else:
                    if merge_peak_response:
                        nonlocal src_peak_response
                        src_peak_response += x[3]
                    return None

            instance_list = list(filter(iou_filter, instance_list))
        return selected_instances

    def instance_seg(self, class_response_maps, peak_list, peak_response_maps, retrieval_cfg):
        # cast tensors to numpy array
        class_response_maps = class_response_maps.squeeze().cpu().numpy()
        peak_list = peak_list.cpu().numpy()
        peak_response_maps = peak_response_maps.cpu().numpy()

        img_height, img_width = peak_response_maps.shape[1], peak_response_maps.shape[2]

        # image size
        img_area = img_height * img_width

        # segment proposals off-the-shelf
        proposals = retrieval_cfg['proposals']

        # proposal contour width
        contour_width = retrieval_cfg.get('contour_width', 5)

        # limit range of proposal size
        proposal_size_limit = retrieval_cfg.get('proposal_size_limit', (0.00002, 0.85))

        # selected number of proposals
        proposal_count = retrieval_cfg.get('proposal_count', 100)

        # nms threshold
        nms_threshold = retrieval_cfg.get('nms_threshold', 0.3)

        # merge peak response during nms
        merge_peak_response = retrieval_cfg.get('merge_peak_response', True)

        # metric free parameters
        param = retrieval_cfg.get('param', None)

        # process each peak
        instance_list = []
        for i in range(len(peak_response_maps)):
            class_idx = peak_list[i, 1]

            # extract hyper-params
            if isinstance(param, tuple):
                # shared param
                bg_threshold_factor, penalty_factor, balance_factor = param
            elif isinstance(param, list):
                # independent params between classes
                bg_threshold_factor, penalty_factor, balance_factor = param[class_idx]
            else:
                raise TypeError('Invalid hyper-params "%s".' % param)

            class_response = imresize(class_response_maps[class_idx], (img_height, img_width), interp='bicubic')
            bg_response = (class_response < bg_threshold_factor * class_response.mean()).astype(np.float32)
            peak_response_map = peak_response_maps[i]

            # select proposal
            max_val = -np.inf
            instance_mask = None

            for j in range(min(proposal_count, len(proposals))):
                raw_mask = imresize(proposals[j].astype(int), peak_response_map.shape, interp='nearest')
                # get contour of the proposal
                contour_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_GRADIENT,
                                                np.ones((contour_width, contour_width), np.uint8)).astype(bool)
                mask = raw_mask.astype(bool)
                # metric
                mask_area = mask.sum()
                if (mask_area >= proposal_size_limit[1] * img_area) or \
                        (mask_area < proposal_size_limit[0] * img_area):
                    continue
                else:
                    val = balance_factor * peak_response_map[mask].sum() + \
                          peak_response_map[contour_mask].sum() - \
                          penalty_factor * bg_response[mask].sum()
                    if val > max_val:
                        max_val = val
                        instance_mask = mask

            if instance_mask is not None:
                instance_list.append((max_val, class_idx, instance_mask, peak_response_map))

        instance_list = sorted(instance_list, key=lambda x: x[0], reverse=True)
        if nms_threshold is not None:
            instance_list = self.instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), nms_threshold,
                                              merge_peak_response)
        return [dict(score=v[0], category=v[1], mask=v[2], prm=v[3]) for v in instance_list]

    def forward(self, input, lables, class_threshold=0, peak_threshold=10, retrieval_cfg=None):
        assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
        if self.inferencing:
            input.requires_grad_()

        # classification network forwarding

        class_response_maps = super(PeakResponseMapping, self).forward(input)

        if self.enable_peak_stimulation:
            # sub-pixel peak finding
            if self.sub_pixel_locating_factor > 1:
                class_response_maps = F.upsample(class_response_maps, scale_factor=self.sub_pixel_locating_factor,
                                                 mode='bilinear', align_corners=True)
            # aggregate responses from informative receptive fields estimated via class peak responses
            peak_list, aggregation = peak_stimulation(class_response_maps, win_size=self.win_size,
                                                      peak_filter=self.peak_filter)
        else:
            # aggregate responses from all receptive fields
            peak_list, aggregation = None, F.adaptive_avg_pool2d(class_response_maps, 1).squeeze(2).squeeze(2)

        if self.inferencing:
            if not self.enable_peak_backprop:
                # extract only class-aware visual cues
                return aggregation, class_response_maps

            # extract instance-aware visual cues, i.e., peak response maps
            assert class_response_maps.size(
                0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'
            if peak_list is None:
                peak_list = peak_stimulation(class_response_maps, return_aggregation=False, win_size=self.win_size,
                                             peak_filter=self.peak_filter)

            peak_response_maps = []
            valid_peak_list = []
            peak_score = []
            # peak backpropagation
            grad_output = class_response_maps.new_empty(class_response_maps.size())

            # print(peak_list)
            # for each gt label
            for cls_idx in lables:
                peak_list_cls = peak_list[peak_list[:, 1] == cls_idx]
                # print(peak_list_cls)
                exist = 0
                for idx in range(peak_list_cls.size(0)):
                    peak_val = class_response_maps[
                        peak_list_cls[idx, 0], peak_list_cls[idx, 1], peak_list_cls[idx, 2], peak_list_cls[idx, 3]]
                    # print(peak_val)
                    if peak_val > peak_threshold:
                        exist = 1
                        grad_output.zero_()
                        # starting from the peak
                        grad_output[peak_list_cls[idx, 0], peak_list_cls[idx, 1], peak_list_cls[idx, 2], peak_list_cls[
                            idx, 3]] = 1
                        if input.grad is not None:
                            input.grad.zero_()
                        class_response_maps.backward(grad_output, retain_graph=True)
                        prm = input.grad.detach().sum(1).clone().clamp(min=0)
                        peak_response_maps.append(prm / prm.sum())
                        valid_peak_list.append(peak_list_cls[idx, :])
                        peak_score.append(peak_val)
                if exist == 0:
                    peak_val_list = [class_response_maps[0, cls_idx, peak_list_cls[idx, 2], peak_list_cls[idx, 3]] for
                                     idx in range(peak_list_cls.size(0))]
                    peak_val_list = torch.tensor(peak_val_list)
                    peak_val = class_response_maps[peak_list_cls[peak_val_list.argmax(), 0], \
                                                   peak_list_cls[peak_val_list.argmax(), 1], \
                                                   peak_list_cls[peak_val_list.argmax(), 2], \
                                                   peak_list_cls[peak_val_list.argmax(), 3]]
                    grad_output.zero_()
                    # starting from the peak
                    grad_output[peak_list_cls[peak_val_list.argmax(), 0], \
                                peak_list_cls[peak_val_list.argmax(), 1], \
                                peak_list_cls[peak_val_list.argmax(), 2], \
                                peak_list_cls[peak_val_list.argmax(), 3]] = 1
                    if input.grad is not None:
                        input.grad.zero_()
                    class_response_maps.backward(grad_output, retain_graph=True)
                    prm = input.grad.detach().sum(1).clone().clamp(min=0)
                    peak_response_maps.append(prm / prm.sum())
                    valid_peak_list.append(peak_list_cls[peak_val_list.argmax(), :])
                    peak_score.append(peak_val)

            # return results
            class_response_maps = class_response_maps.detach()
            aggregation = aggregation.detach()

            if len(peak_response_maps) > 0:
                valid_peak_list = torch.stack(valid_peak_list)
                peak_response_maps = torch.cat(peak_response_maps, 0)
                peak_score = torch.tensor(peak_score)
                if retrieval_cfg is None:
                    # classification confidence scores, class-aware and instance-aware visual cues
                    return aggregation, class_response_maps, valid_peak_list, peak_response_maps, peak_score
                else:
                    # instance segmentation using build-in proposal retriever
                    return self.instance_seg(class_response_maps, valid_peak_list, peak_response_maps, retrieval_cfg)
            else:
                return None  # aggregation, class_response_maps # None
        else:
            # classification confidence scores
            return aggregation, peak_list, class_response_maps

    def train(self, mode=True):
        super(PeakResponseMapping, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super(PeakResponseMapping, self).train(False)
        self._patch()
        self.inferencing = True
        return self


# %%
class FC_ResNet(nn.Module):

    def __init__(self, model, num_classes):
        super(FC_ResNet, self).__init__()

        # feature encoding
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classifier
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def fc_resnet50(num_classes: int = 20, pretrained: bool = True) -> nn.Module:
    """FC ResNet50.
    """
    model = FC_ResNet(models.resnet50(pretrained), num_classes)
    return model


def peak_response_mapping(
        backbone: nn.Module = fc_resnet50(),
        enable_peak_stimulation: bool = True,
        enable_peak_backprop: bool = True,
        win_size: int = 3,
        sub_pixel_locating_factor: int = 8,
        filter_type: Union[str, int, float] = 'median') -> nn.Module:
    """Peak Response Mapping.
    """

    model = PeakResponseMapping(
        backbone,
        enable_peak_stimulation=enable_peak_stimulation,
        enable_peak_backprop=enable_peak_backprop,
        win_size=win_size,
        sub_pixel_locating_factor=sub_pixel_locating_factor,
        filter_type=filter_type)
    return model


voc_id_name_map = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
                   9: 'chair', 10: 'cow',
                   11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant',
                   17: 'sheep', 18: 'sofa',
                   19: 'train', 20: 'tvmonitor'}


def imresize(arr, size, interp='bilibear', mode=None):
    im = Image.fromarray(np.uint8(arr), mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size) * percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'biliear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])  # 调用PIL库中的resize函数
    return np.array(imnew)


# copy from IAM
# get low level feature
class FC_Resnet_aff(nn.Module):
    def __init__(self, model, num_classes):
        super(FC_Resnet_aff, self).__init__()

        # feature encoding
        self.res_block1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool)

        self.res_block2 = model.layer1
        self.res_block3 = model.layer2
        self.res_block4 = model.layer3
        self.res_block5 = model.layer4

        # classifier
        self.num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(self.num_features, num_classes, kernel_size=1, bias=True))

    def forward(self, x):
        p1 = self.res_block1(x)
        p2 = self.res_block2(p1)
        p3 = self.res_block3(p2)
        p4 = self.res_block4(p3)
        p5 = self.res_block5(p4)

        class_response_maps = self.classifier(p5)

        return class_response_maps, p1, p2, p3, p4


def fc_resnet50_aff(num_classes: int = 20, pretrained: bool = True) -> nn.Module:
    model = FC_Resnet_aff(models.resnet50(pretrained), num_classes)
    return model


def peak_response_mapping_aff(
        backbone: nn.Module = fc_resnet50_aff(),
        enable_peak_stimulation: bool = True,
        enable_peak_backprop: bool = True,
        win_size: int = 3,
        sub_pixel_locating_factor: int = 8,
        filter_type: Union[str, int, float] = 'median') -> nn.Module:
    """Peak Response Mapping.
    """

    model = PeakResponseMapping_aff(
        backbone,
        enable_peak_stimulation=enable_peak_stimulation,
        enable_peak_backprop=enable_peak_backprop,
        win_size=win_size,
        sub_pixel_locating_factor=sub_pixel_locating_factor,
        filter_type=filter_type)
    return model


# only for inferencing
class PeakResponseMapping_aff(nn.Sequential):

    def __init__(self, *args, **kargs):
        super(PeakResponseMapping_aff, self).__init__(*args)

        self.inferencing = True
        # use global average pooling to aggregate responses if peak stimulation is disabled
        self.enable_peak_stimulation = kargs.get('enable_peak_stimulation', True)
        # return only the class response maps in inference mode if peak backpropagation is disabled
        self.enable_peak_backprop = kargs.get('enable_peak_backprop', True)
        # window size for peak finding
        self.win_size = kargs.get('win_size', 3)
        # sub-pixel peak finding
        self.sub_pixel_locating_factor = kargs.get('sub_pixel_locating_factor', 1)
        # peak filtering
        self.filter_type = kargs.get('filter_type', 'median')
        if self.filter_type == 'median':
            self.peak_filter = self._median_filter
        elif self.filter_type == 'mean':
            self.peak_filter = self._mean_filter
        elif self.filter_type == 'max':
            self.peak_filter = self._max_filter
        elif isinstance(self.filter_type, (int, float)):
            self.peak_filter = lambda x: self.filter_type
        else:
            self.peak_filter = None

    @staticmethod
    def _median_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _mean_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _max_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _lzc_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return 0.5 * threshold.contiguous().view(batch_size, num_channels, 1, 1)

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward

    def instance_nms(self, instance_list, threshold=0.3, merge_peak_response=True):
        selected_instances = []
        while len(instance_list) > 0:
            instance = instance_list.pop(0)
            selected_instances.append(instance)
            src_mask = instance[2].astype(bool)
            src_peak_response = instance[3]

            def iou_filter(x):
                dst_mask = x[2].astype(bool)
                # IoU
                intersection = np.logical_and(src_mask, dst_mask).sum()
                union = np.logical_or(src_mask, dst_mask).sum()
                iou = intersection / (union + 1e-10)
                if iou < threshold:
                    return x
                else:
                    if merge_peak_response:
                        nonlocal src_peak_response
                        src_peak_response += x[3]
                    return None

            instance_list = list(filter(iou_filter, instance_list))
        return selected_instances

    def instance_seg(self, class_response_maps, peak_list, peak_response_maps, retrieval_cfg):
        # cast tensors to numpy array
        class_response_maps = class_response_maps.squeeze().cpu().numpy()
        peak_list = peak_list.cpu().numpy()
        peak_response_maps = peak_response_maps.cpu().numpy()

        img_height, img_width = peak_response_maps.shape[1], peak_response_maps.shape[2]

        # image size
        img_area = img_height * img_width

        # segment proposals off-the-shelf
        proposals = retrieval_cfg['proposals']

        # proposal contour width
        contour_width = retrieval_cfg.get('contour_width', 5)

        # limit range of proposal size
        proposal_size_limit = retrieval_cfg.get('proposal_size_limit', (0.00002, 0.85))

        # selected number of proposals
        proposal_count = retrieval_cfg.get('proposal_count', 100)

        # nms threshold
        nms_threshold = retrieval_cfg.get('nms_threshold', 0.3)

        # merge peak response during nms
        merge_peak_response = retrieval_cfg.get('merge_peak_response', True)

        # metric free parameters
        param = retrieval_cfg.get('param', None)

        # process each peak
        instance_list = []
        for i in range(len(peak_response_maps)):
            class_idx = peak_list[i, 1]

            # extract hyper-params
            if isinstance(param, tuple):
                # shared param
                bg_threshold_factor, penalty_factor, balance_factor = param
            elif isinstance(param, list):
                # independent params between classes
                bg_threshold_factor, penalty_factor, balance_factor = param[class_idx]
            else:
                raise TypeError('Invalid hyper-params "%s".' % param)

            class_response = imresize(class_response_maps[class_idx], (img_height, img_width), interp='bicubic')
            bg_response = (class_response < bg_threshold_factor * class_response.mean()).astype(np.float32)
            peak_response_map = peak_response_maps[i]

            # select proposal
            max_val = -np.inf
            instance_mask = None

            for j in range(min(proposal_count, len(proposals))):
                raw_mask = imresize(proposals[j].astype(int), peak_response_map.shape, interp='nearest')
                # get contour of the proposal
                contour_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_GRADIENT,
                                                np.ones((contour_width, contour_width), np.uint8)).astype(bool)
                mask = raw_mask.astype(bool)
                # metric
                mask_area = mask.sum()
                if (mask_area >= proposal_size_limit[1] * img_area) or \
                        (mask_area < proposal_size_limit[0] * img_area):
                    continue
                else:
                    val = balance_factor * peak_response_map[mask].sum() + \
                          peak_response_map[contour_mask].sum() - \
                          penalty_factor * bg_response[mask].sum()
                    if val > max_val:
                        max_val = val
                        instance_mask = mask

            if instance_mask is not None:
                instance_list.append((max_val, class_idx, instance_mask, peak_response_map))

        instance_list = sorted(instance_list, key=lambda x: x[0], reverse=True)
        if nms_threshold is not None:
            instance_list = self.instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), nms_threshold,
                                              merge_peak_response)
        return [dict(score=v[0], category=v[1], mask=v[2], prm=v[3]) for v in instance_list]

    # # cam normalize to RW
    # def forward(self, input, lables, img,class_threshold=0, peak_threshold=10, retrieval_cfg=None,graph_op=None,resolution=7):
    #     assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
    #     dir = "./voc_visual"
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #     with torch.no_grad():
    #
    #         # _, _, H, W = input.shape  # batchsize == 1
    #         H,W,_ = np.array(img).shape
    #
    #         LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #         HSV_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #
    #         # classification network forwarding
    #         class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
    #
    #         _, cls_num, cam_H, cam_W = class_response_maps.shape
    #
    #         onehot_label = torch.tensor(self.label2onehot(lables,cls_num))
    #         class_response_maps = class_response_maps * torch.tensor(onehot_label).view(1,cls_num,1,1)
    #
    #         class_response_maps = class_response_maps.detach()
    #         class_response_maps_normalize = class_response_maps.numpy()
    #         class_response_maps_normalize = normalize_cam(class_response_maps_normalize)
    #
    #         class_response_maps_normalize = torch.tensor(class_response_maps_normalize)
    #         org_size_class_response_maps = F.upsample(class_response_maps_normalize, size=(H,W), mode='bilinear', align_corners=True)
    #
    #         # use p2 as feature
    #         feature = p4
    #
    #         _, _, f_H, f_W = feature.shape
    #
    #         # spatial_scale
    #         spatial_scale = f_H / H
    #         Align_op = RoIAlign((resolution, resolution), spatial_scale, 0)
    #
    #         # graph operate
    #         sp_label = graph_op["sp_label"]
    #         adjacent_matrix = graph_op["adjacent_matrix"]
    #         img_id = graph_op["img_id"]
    #         belong_mat = graph_op["belong_mat"]
    #         proposal_score = graph_op["proposal_score"]
    #         print(img_id)
    #
    #         # add bg_score
    #         bg_score = 1 - proposal_score.sum(-1)
    #         proposal_score = np.concatenate((bg_score[:,None],proposal_score),axis=1)
    #
    #         sp_scores = np.zeros((adjacent_matrix.shape[0],proposal_score.shape[-1])) # sp_num, cls_num + 1
    #         for row_idx in range(belong_mat.shape[0]):
    #             score = proposal_score[row_idx]
    #             belong_row = belong_mat[row_idx] != 0
    #
    #             sp_scores[belong_row] += score
    #
    #         aff_proposal_score = torch.tensor(sp_scores)
    #         low_label = np.min(sp_label)
    #         high_label = np.max(sp_label)
    #
    #         for idx, temp_label in tqdm(enumerate(range(low_label, high_label + 1)),total=adjacent_matrix.shape[0]):
    #             sp_mask = sp_label == temp_label
    #
    #             sp_mask_tensor = torch.tensor(sp_mask)
    #             cam_scores = (sp_mask_tensor * org_size_class_response_maps).sum((-2,-1)) / sp_mask_tensor.sum() # --> (1, cls_num)
    #
    #             ind_xy = np.nonzero(sp_mask)
    #             xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
    #
    #             mask = sp_mask[ymin:ymax, xmin:xmax]
    #             mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
    #             mask = torch.tensor(mask)
    #
    #             box = torch.tensor([[0, xmin, ymin, xmax, ymax]],dtype=torch.float32)
    #             box_feature = Align_op(feature, box)  # (1,7,7)
    #
    #             sp_f = (mask * box_feature).view(-1, resolution, resolution)  # (1,dim,7,7) --> (dim,7,7)
    #             sp_f = sp_f.sum((-2,-1))  # (dim)
    #             sp_f = sp_f / (mask.sum()+1e-5)
    #
    #             sp_f = sp_f[None, :]
    #
    #             if idx == 0:
    #                 sp_cams = cam_scores
    #                 sp_features = sp_f
    #
    #                 color_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                 for color_idx in range(1,3):
    #                     temp_feature = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     color_feature = np.concatenate((color_feature,temp_feature),axis=0)
    #
    #                 rgb_features = color_feature.transpose()
    #
    #             else:
    #                 sp_cams = torch.cat((sp_cams, cam_scores), dim=0)
    #                 sp_features = torch.cat((sp_features, sp_f), dim=0)
    #
    #                 color_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                 for color_idx in range(1,3):
    #                     temp_feature = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     color_feature = np.concatenate((color_feature,temp_feature),axis=0)
    #
    #                 rgb_features = np.concatenate((rgb_features,color_feature.transpose()),axis=0)
    #
    #         rgb_features = torch.tensor(rgb_features)
    #
    #         # make similarity
    #         aff_sp_f = torch.nn.functional.cosine_similarity(sp_features.unsqueeze(0), sp_features.unsqueeze(1), dim=-1)
    #         aff_score = torch.nn.functional.cosine_similarity(aff_proposal_score.unsqueeze(0), aff_proposal_score.unsqueeze(1), dim=-1)
    #         aff_rgb = torch.nn.functional.cosine_similarity(rgb_features.unsqueeze(0), rgb_features.unsqueeze(1), dim=-1)
    #         # belong_mat = torch.tensor(belong_mat).t()
    #         # aff_belong = torch.nn.functional.cosine_similarity(belong_mat.unsqueeze(0),belong_mat.unsqueeze(1),dim=-1)
    #
    #         aff_mat = aff_sp_f + aff_score + aff_rgb
    #
    #         aff_mat_mask = aff_mat * (torch.tensor(adjacent_matrix, dtype=torch.float32) + torch.eye(adjacent_matrix.shape[0]))
    #
    #         trans_mat = aff_mat_mask / (torch.sum(aff_mat_mask, dim=0, keepdim=True)+1e-5)
    #
    #         # logt = 4
    #         #
    #         # for _ in range(logt):
    #         #     trans_mat = torch.matmul(trans_mat, trans_mat)
    #
    #         # trans_mat = torch.linalg.inv(1e-5 * torch.eye(aff_mat.shape[0]) + torch.tensor(aff_mat_mask,dtype=torch.float32)) @ torch.tensor(aff_mat_mask,dtype=torch.float32)
    #
    #         plt.close()
    #         fig,axx = plt.subplots(2, 3)
    #
    #         sns.heatmap(aff_sp_f.numpy(),cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,0])
    #         axx[0,0].set_title("aff_sp_f")
    #
    #         sns.heatmap(aff_score.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,1])
    #         axx[0,1].set_title("aff_score")
    #
    #         # axx[0,2].axis("off")
    #         sns.heatmap(aff_rgb.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,2])
    #         axx[0,2].set_title("aff_rgb")
    #
    #         sns.heatmap(aff_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,0])
    #         axx[1,0].set_title("aff_mat")
    #
    #         sns.heatmap(aff_mat_mask.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,1])
    #         axx[1,1].set_title("aff_mat_mask")
    #
    #         sns.heatmap(trans_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,2])
    #         axx[1,2].set_title("trans_mat")
    #
    #         print(os.path.join(dir,str(img_id)+"_aff.jpg"))
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+"_aff.jpg"), dpi=300)
    #
    #         sp_cams = sp_cams.numpy() - sp_cams.numpy().min(0)
    #         sp_cams = sp_cams / (1e-5 + sp_cams.max(0))
    #         sp_cams = torch.tensor(sp_cams)
    #         # print(sp_cams)
    #
    #         RW_cams = torch.tensor(sp_cams.t(), dtype=torch.float32)
    #         trans_mat = torch.tensor(trans_mat, dtype=torch.float32)
    #
    #         cam_valid_peak_list, cam_peak_score = self.get_peak_list(class_response_maps,lables,peak_threshold)
    #         response_maps = [class_response_maps]
    #         valid_peak_list = [cam_valid_peak_list]
    #         scale = 0.7
    #
    #         iter_TIME = 4
    #         alph = 0.7
    #
    #         for _ in range(iter_TIME):
    #             RW_cams = alph * torch.matmul(RW_cams, trans_mat) + (1 - alph) * torch.tensor(sp_cams.t(),dtype=torch.float32)
    #             RW_cams = RW_cams.numpy() - RW_cams.numpy().min(0)
    #             RW_cams = RW_cams / (1e-5 + RW_cams.max(0))
    #             RW_cams = torch.tensor(RW_cams)
    #
    #             RW_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_cams.dtype)
    #
    #             for cls_idx in lables:
    #                 for idx, temp_label in enumerate(range(low_label, high_label + 1)):
    #                     sp_mask = sp_label == temp_label
    #                     sp_mask_tensor = torch.tensor(sp_mask)
    #                     RW_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_cams[cls_idx,temp_label] # single batch input
    #
    #             RW_class_response_maps = F.upsample(RW_class_response_maps,size=(cam_H,cam_W),mode='bilinear', align_corners=True)
    #
    #             RW_valid_peak_list, RW_peak_score = self.get_peak_list(RW_class_response_maps,lables,0.9)
    #
    #             response_maps.append(RW_class_response_maps)
    #             valid_peak_list.append(RW_valid_peak_list)
    #
    #         cam_num = onehot_label.sum().item()
    #         plt.close()
    #         f, axarr = plt.subplots(len(response_maps), int(cam_num + 1))
    #         plt.tight_layout()
    #
    #         for idx in range(len(response_maps)):
    #             if idx == 0:
    #                 axarr[idx, 0].imshow(img)
    #             else:
    #                 axarr[idx, 0].imshow(mark_boundaries(img, sp_label))
    #
    #             axarr[idx, 0].axis("off")
    #
    #             for i,cls_idx in enumerate(lables):
    #                 CRM = response_maps[idx]
    #
    #                 cam = CRM[0, cls_idx].detach().cpu().numpy()
    #                 cam = cam - np.min(cam)
    #                 cam = cam / np.max(cam)
    #                 cam = np.uint8(255 * cam)
    #
    #                 heatmap = cv2.applyColorMap(cv2.resize(cam, (W, H)), cv2.COLORMAP_JET)
    #                 heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #                 result = heatmap * scale + (img) * (1 - scale)
    #                 result = np.uint8(result)
    #                 result = cv2.resize(result, (int(cam_W*self.sub_pixel_locating_factor),int(cam_H*self.sub_pixel_locating_factor)))
    #
    #                 axarr[idx, i+1].imshow(result)
    #                 axarr[idx, i+1].axis("off")
    #                 axarr[idx, i+1].set_title(voc_id_name_map[cls_idx.item()+1])
    #
    #                 VPL = valid_peak_list[idx]
    #
    #                 for peak in VPL:
    #                     if peak[1] == cls_idx:
    #                         axarr[idx, i+1].scatter(peak[3],peak[2], color='purple',marker='*',edgecolors='purple', s=17)
    #
    #                 axarr[idx, i+1].axis("off")
    #
    #         # plt.show()
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+".jpg"), dpi=300)
    #
    #         # if len(peak_score) > 0:
    #         #     valid_peak_list = torch.stack(valid_peak_list)
    #         #     peak_score = torch.tensor(peak_score)
    #         #     # classification confidence scores, class-aware and instance-aware visual cues
    #         #     return class_response_maps, valid_peak_list, peak_score
    #         # else:
    #         #     return None

    # # use proposal score to replace cam score
    # # draw
    # def forward(self, input, lables, img,class_threshold=0, peak_threshold=10, retrieval_cfg=None,graph_op=None,resolution=7):
    #     assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
    #     dir = "./voc_visual"
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #     with torch.no_grad():
    #
    #         # _, _, H, W = input.shape  # batchsize == 1
    #         H,W,_ = np.array(img).shape
    #
    #         LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #         HSV_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #
    #         # classification network forwarding
    #         class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
    #         class_response_maps = class_response_maps.detach()
    #         org_size_class_response_maps = F.upsample(class_response_maps, size=(H,W), mode='bilinear', align_corners=True)
    #
    #         _, cls_num, cam_H, cam_W = class_response_maps.shape
    #
    #         onehot_label = self.label2onehot(lables,cls_num)
    #
    #         _, _, f_H_p1, f_W_p1 = p1.shape
    #         _, _, f_H_p2, f_W_p2 = p2.shape
    #         _, _, f_H_p3, f_W_p3 = p3.shape
    #         _, _, f_H_p4, f_W_p4 = p4.shape
    #
    #         Align_op_p1 = RoIAlign((resolution, resolution), f_H_p1/H, 0)
    #         Align_op_p2 = RoIAlign((resolution, resolution), f_H_p2/H, 0)
    #         Align_op_p3 = RoIAlign((resolution, resolution), f_H_p3/H, 0)
    #         Align_op_p4 = RoIAlign((resolution, resolution), f_H_p4/H, 0)
    #
    #         # graph operate
    #         sp_label = graph_op["sp_label"]
    #         adjacent_matrix = graph_op["adjacent_matrix"]
    #         img_id = graph_op["img_id"]
    #         belong_mat = graph_op["belong_mat"]
    #         proposal_score = graph_op["proposal_score"]
    #         print(img_id)
    #
    #         # add bg_score
    #         bg_score = 1 - proposal_score.sum(-1)
    #         proposal_score = np.concatenate((bg_score[:,None],proposal_score),axis=1)
    #         # proposal_score = np.concatenate((bg_score[:,None],proposal_score*onehot_label),axis=1)
    #
    #         # # lzc edit 9-6
    #         # max_idx = np.argmax(proposal_score,axis=-1)
    #         # flag = np.zeros_like(max_idx)
    #         # for label in lables:
    #         #     flag += (label+1).item() == max_idx
    #         #
    #         # bg_score_const = np.zeros(proposal_score.shape[-1])
    #         # bg_score_const[0] = 1
    #         # proposal_score[flag == 0] = bg_score_const
    #         # print(flag)
    #
    #         sp_scores = np.zeros((adjacent_matrix.shape[0],proposal_score.shape[-1])) # sp_num, cls_num + 1
    #         for row_idx in range(belong_mat.shape[0]):
    #             score = proposal_score[row_idx]
    #             belong_row = belong_mat[row_idx] != 0
    #
    #             sp_scores[belong_row] += score
    #
    #         sp_scores = torch.tensor(sp_scores)
    #         sp_scores_normalize = torch.nn.functional.normalize(sp_scores,p=1,dim=-1)
    #
    #         low_label = np.min(sp_label)
    #         high_label = np.max(sp_label)
    #
    #         for idx, temp_label in tqdm(enumerate(range(low_label, high_label + 1)),total=adjacent_matrix.shape[0]):
    #             sp_mask = sp_label == temp_label
    #             if idx == 0:
    #                 sp_proposal = sp_mask[None,:,:]
    #             else:
    #                 sp_proposal = np.concatenate((sp_proposal, sp_mask[None,:,:]),axis=0)
    #
    #             # sp_mask_tensor = torch.tensor(sp_mask)
    #
    #             ind_xy = np.nonzero(sp_mask)
    #             xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
    #
    #             mask = sp_mask[ymin:ymax, xmin:xmax]
    #             mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
    #             mask = torch.tensor(mask)
    #
    #             box = torch.tensor([[0, xmin, ymin, xmax, ymax]],dtype=torch.float32)
    #             box_feature_p1 = Align_op_p1(p1, box)  # (1,7,7)
    #             box_feature_p2 = Align_op_p2(p2, box)  # (1,7,7)
    #             box_feature_p3 = Align_op_p3(p3, box)  # (1,7,7)
    #             box_feature_p4 = Align_op_p4(p4, box)  # (1,7,7)
    #
    #             sp_feature_p1 = get_sp_f(box_feature_p1,mask,resolution)
    #             sp_feature_p2 = get_sp_f(box_feature_p2,mask,resolution)
    #             sp_feature_p3 = get_sp_f(box_feature_p3,mask,resolution)
    #             sp_feature_p4 = get_sp_f(box_feature_p4,mask,resolution)
    #
    #             if idx == 0:
    #                 sp_features_p1 = sp_feature_p1
    #                 sp_features_p2 = sp_feature_p2
    #                 sp_features_p3 = sp_feature_p3
    #                 sp_features_p4 = sp_feature_p4
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1,3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature,temp_feature_rgb),axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature,temp_feature_lab),axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature,temp_feature_hsv),axis=0)
    #
    #                 RGB_features = RGB_feature.transpose()
    #                 LAB_features = LAB_feature.transpose()
    #                 HSV_features = HSV_feature.transpose()
    #
    #             else:
    #                 sp_features_p1 = torch.cat((sp_features_p1, sp_feature_p1),dim=0)
    #                 sp_features_p2 = torch.cat((sp_features_p2, sp_feature_p2),dim=0)
    #                 sp_features_p3 = torch.cat((sp_features_p3, sp_feature_p3),dim=0)
    #                 sp_features_p4 = torch.cat((sp_features_p4, sp_feature_p4),dim=0)
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1, 3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature, temp_feature_rgb), axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature, temp_feature_lab), axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature, temp_feature_hsv), axis=0)
    #
    #                 RGB_features = np.concatenate((RGB_features,RGB_feature.transpose()),axis=0)
    #                 LAB_features = np.concatenate((LAB_features,LAB_feature.transpose()),axis=0)
    #                 HSV_features = np.concatenate((HSV_features,HSV_feature.transpose()),axis=0)
    #
    #         RGB_features = torch.tensor(RGB_features)
    #         LAB_features = torch.tensor(LAB_features)
    #         HSV_features = torch.tensor(HSV_features)
    #
    #         # make similarity
    #         aff_sp_f = torch.nn.functional.cosine_similarity(sp_features_p1.unsqueeze(0), sp_features_p1.unsqueeze(1), dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p2.unsqueeze(0), sp_features_p2.unsqueeze(1),dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p3.unsqueeze(0), sp_features_p3.unsqueeze(1),dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p4.unsqueeze(0), sp_features_p4.unsqueeze(1),dim=-1)
    #         # aff_sp_f = torch.nn.functional.cosine_similarity(sp_features_p4.unsqueeze(0), sp_features_p4.unsqueeze(1),dim=-1)
    #
    #         aff_score = torch.nn.functional.cosine_similarity(sp_scores.unsqueeze(0), sp_scores.unsqueeze(1), dim=-1)
    #
    #         aff_color = torch.nn.functional.cosine_similarity(RGB_features.unsqueeze(0), RGB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(HSV_features.unsqueeze(0), HSV_features.unsqueeze(1), dim=-1)
    #         # aff_color = torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1), dim=-1)
    #
    #         aff_mat = aff_sp_f + aff_color + aff_score
    #         # aff_mat = aff_sp_f * aff_color
    #         # aff_mat = aff_sp_f * aff_score * aff_rgb
    #
    #         aff_mat_mask = aff_mat * (torch.tensor(adjacent_matrix, dtype=torch.float32) + torch.eye(adjacent_matrix.shape[0]))
    #
    #         trans_mat = aff_mat_mask / (torch.sum(aff_mat_mask, dim=0, keepdim=True)+1e-5)
    #
    #         plt.close()
    #         fig,axx = plt.subplots(2, 3)
    #
    #         sns.heatmap(aff_sp_f.numpy(),cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,0])
    #         axx[0,0].set_title("aff_sp_f")
    #
    #         sns.heatmap(aff_score.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,1])
    #         axx[0,1].set_title("aff_score")
    #
    #         # axx[0,2].axis("off")
    #         sns.heatmap(aff_color.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,2])
    #         axx[0,2].set_title("aff_color")
    #
    #         sns.heatmap(aff_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,0])
    #         axx[1,0].set_title("aff_mat")
    #
    #         sns.heatmap(aff_mat_mask.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,1])
    #         axx[1,1].set_title("aff_mat_mask")
    #
    #         sns.heatmap(trans_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,2])
    #         axx[1,2].set_title("trans_mat")
    #
    #         print(os.path.join(dir,str(img_id)+"_aff.jpg"))
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+"_aff.jpg"), dpi=300)
    #
    #         # 包括背景类 --> (21, sp_num)
    #         RW_cams = torch.tensor(sp_scores_normalize.t(), dtype=torch.float32)
    #         trans_mat = torch.tensor(trans_mat, dtype=torch.float32)
    #
    #         cam_valid_peak_list, cam_peak_score = self.get_peak_list(class_response_maps,lables,peak_threshold)
    #         response_maps = [class_response_maps]
    #         valid_peak_list = [cam_valid_peak_list]
    #         scale = 0.7
    #
    #         # RW_cams = torch.matmul(torch.tensor(trans_mat.t(),dtype=torch.float32),torch.tensor(sp_cams,dtype=torch.float32))
    #         # RW_cams = RW_cams.t()
    #
    #         iter_TIME = 4 + 1
    #         alph = 1
    #
    #         for i in range(iter_TIME):
    #             # print(RW_cams.shape)
    #             # print(sp_scores_normalize.shape)
    #             if i == 0:
    #                 RW_cams = RW_cams
    #             else:
    #                 RW_cams = alph * torch.matmul(RW_cams, trans_mat) + (1 - alph) * torch.tensor(sp_scores_normalize.t(),dtype=torch.float32)
    #             RW_cams = torch.nn.functional.normalize(RW_cams, p=1, dim=0)
    #
    #             RW_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_cams.dtype)
    #
    #             for cls_idx in lables:
    #                 for idx, temp_label in enumerate(range(low_label, high_label + 1)):
    #                     sp_mask = sp_label == temp_label
    #                     sp_mask_tensor = torch.tensor(sp_mask)
    #                     # exclusive bg class
    #                     RW_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_cams[cls_idx + 1,idx] # single batch input
    #
    #             RW_class_response_maps = F.upsample(RW_class_response_maps,size=(cam_H,cam_W),mode='bilinear', align_corners=True)
    #
    #             RW_valid_peak_list, RW_peak_score = self.get_peak_list(RW_class_response_maps,lables,0,self._lzc_filter)
    #
    #             response_maps.append(RW_class_response_maps)
    #             valid_peak_list.append(RW_valid_peak_list)
    #
    #         RW_class_response_maps = RW_class_response_maps.numpy()
    #         RW_class_response_maps = RW_class_response_maps - np.min(RW_class_response_maps,(-2,-1),keepdims = True)
    #         RW_class_response_maps = RW_class_response_maps / np.max(RW_class_response_maps,(-2,-1),keepdims = True)
    #         RW_class_response_maps = torch.tensor(RW_class_response_maps)
    #
    #         # print(RW_class_response_maps)
    #         mask_CAM = RW_class_response_maps * class_response_maps
    #         mask_CAM_valid_peak_list, mask_CAM_peak_score = self.get_peak_list(mask_CAM, lables, peak_threshold)
    #         response_maps.append(mask_CAM)
    #         valid_peak_list.append(mask_CAM_valid_peak_list)
    #
    #         cam_num = onehot_label.sum().item()
    #         plt.close()
    #         f, axarr = plt.subplots(len(response_maps), int(cam_num + 1))
    #         plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.1, hspace=0.1)
    #
    #         for idx in range(len(response_maps)):
    #             if idx == 0:
    #                 axarr[idx, 0].imshow(img)
    #             else:
    #                 axarr[idx, 0].imshow(mark_boundaries(img, sp_label))
    #
    #             axarr[idx, 0].axis("off")
    #
    #             for i,cls_idx in enumerate(lables):
    #                 CRM = response_maps[idx]
    #
    #                 cam = CRM[0, cls_idx].detach().cpu().numpy()
    #                 cam = cam - np.min(cam)
    #                 cam = cam / np.max(cam)
    #                 cam = np.uint8(255 * cam)
    #
    #                 heatmap = cv2.applyColorMap(cv2.resize(cam, (W, H)), cv2.COLORMAP_JET)
    #                 heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #                 result = heatmap * scale + (img) * (1 - scale)
    #                 result = np.uint8(result)
    #                 result = cv2.resize(result, (int(cam_W*self.sub_pixel_locating_factor),int(cam_H*self.sub_pixel_locating_factor)))
    #
    #                 axarr[idx, i+1].imshow(result)
    #                 axarr[idx, i+1].axis("off")
    #                 axarr[idx, i+1].set_title(voc_id_name_map[cls_idx.item()+1])
    #
    #                 VPL = valid_peak_list[idx]
    #
    #                 for peak in VPL:
    #                     if peak[1] == cls_idx:
    #                         axarr[idx, i+1].scatter(peak[3],peak[2], color='purple',marker='*',edgecolors='purple', s=17)
    #
    #                 axarr[idx, i+1].axis("off")
    #
    #         # plt.show()
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+".jpg"), dpi=300)
    #
    #         # if len(peak_score) > 0:
    #         #     valid_peak_list = torch.stack(valid_peak_list)
    #         #     peak_score = torch.tensor(peak_score)
    #         #     # classification confidence scores, class-aware and instance-aware visual cues
    #         #     return class_response_maps, valid_peak_list, peak_score
    #         # else:
    #         #     return None

    # # use proposal score to replace cam score
    # # label assign
    # def forward(self, input, lables, img,class_threshold=0, peak_threshold=10, retrieval_cfg=None,graph_op=None,resolution=7):
    #     assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
    #     with torch.no_grad():
    #         device = input.device
    #
    #         # _, _, H, W = input.shape  # batchsize == 1
    #         H,W,_ = np.array(img).shape
    #
    #         LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #         HSV_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #
    #         # classification network forwarding
    #         class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
    #         class_response_maps = class_response_maps.detach()
    #         org_size_class_response_maps = F.upsample(class_response_maps, size=(H,W), mode='bilinear', align_corners=True)
    #
    #         _, cls_num, cam_H, cam_W = class_response_maps.shape
    #
    #         onehot_label = self.label2onehot(lables,cls_num)
    #
    #         _, _, f_H_p1, f_W_p1 = p1.shape
    #         _, _, f_H_p2, f_W_p2 = p2.shape
    #         _, _, f_H_p3, f_W_p3 = p3.shape
    #         _, _, f_H_p4, f_W_p4 = p4.shape
    #
    #         Align_op_p1 = RoIAlign((resolution, resolution), f_H_p1/H, 0)
    #         Align_op_p2 = RoIAlign((resolution, resolution), f_H_p2/H, 0)
    #         Align_op_p3 = RoIAlign((resolution, resolution), f_H_p3/H, 0)
    #         Align_op_p4 = RoIAlign((resolution, resolution), f_H_p4/H, 0)
    #
    #         # graph operate
    #         sp_label = graph_op["sp_label"]
    #         adjacent_matrix = graph_op["adjacent_matrix"]
    #         img_id = graph_op["img_id"]
    #         belong_mat = graph_op["belong_mat"]
    #         proposal_score = graph_op["proposal_score"]
    #
    #         # add bg_score
    #         bg_score = 1 - proposal_score.sum(-1)
    #         proposal_score = np.concatenate((bg_score[:,None],proposal_score),axis=1)
    #
    #         sp_scores = np.zeros((adjacent_matrix.shape[0],proposal_score.shape[-1])) # sp_num, cls_num + 1
    #         for row_idx in range(belong_mat.shape[0]):
    #             score = proposal_score[row_idx]
    #             belong_row = belong_mat[row_idx] != 0
    #
    #             sp_scores[belong_row] += score
    #
    #         sp_scores = torch.tensor(sp_scores).to(device)
    #         sp_scores_normalize = torch.nn.functional.normalize(sp_scores,p=1,dim=-1)
    #
    #         low_label = np.min(sp_label)
    #         high_label = np.max(sp_label)
    #
    #
    #         for idx, temp_label in enumerate(range(low_label, high_label + 1)):
    #             sp_mask = sp_label == temp_label
    #
    #             # sp_mask_tensor = torch.tensor(sp_mask)
    #
    #             ind_xy = np.nonzero(sp_mask)
    #             xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
    #
    #             mask = sp_mask[ymin:ymax, xmin:xmax]
    #             mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
    #             mask = torch.tensor(mask).to(device)
    #
    #             box = torch.tensor([[0, xmin, ymin, xmax, ymax]],dtype=torch.float32).to(device)
    #             box_feature_p1 = Align_op_p1(p1, box)  # (1,7,7)
    #             box_feature_p2 = Align_op_p2(p2, box)  # (1,7,7)
    #             box_feature_p3 = Align_op_p3(p3, box)  # (1,7,7)
    #             box_feature_p4 = Align_op_p4(p4, box)  # (1,7,7)
    #
    #             sp_feature_p1 = get_sp_f(box_feature_p1,mask,resolution)
    #             sp_feature_p2 = get_sp_f(box_feature_p2,mask,resolution)
    #             sp_feature_p3 = get_sp_f(box_feature_p3,mask,resolution)
    #             sp_feature_p4 = get_sp_f(box_feature_p4,mask,resolution)
    #
    #             if idx == 0:
    #                 sp_features_p1 = sp_feature_p1
    #                 sp_features_p2 = sp_feature_p2
    #                 sp_features_p3 = sp_feature_p3
    #                 sp_features_p4 = sp_feature_p4
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1,3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature,temp_feature_rgb),axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature,temp_feature_lab),axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature,temp_feature_hsv),axis=0)
    #
    #                 RGB_features = RGB_feature.transpose()
    #                 LAB_features = LAB_feature.transpose()
    #                 HSV_features = HSV_feature.transpose()
    #
    #             else:
    #                 sp_features_p1 = torch.cat((sp_features_p1, sp_feature_p1),dim=0)
    #                 sp_features_p2 = torch.cat((sp_features_p2, sp_feature_p2),dim=0)
    #                 sp_features_p3 = torch.cat((sp_features_p3, sp_feature_p3),dim=0)
    #                 sp_features_p4 = torch.cat((sp_features_p4, sp_feature_p4),dim=0)
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1, 3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature, temp_feature_rgb), axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature, temp_feature_lab), axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature, temp_feature_hsv), axis=0)
    #
    #                 RGB_features = np.concatenate((RGB_features,RGB_feature.transpose()),axis=0)
    #                 LAB_features = np.concatenate((LAB_features,LAB_feature.transpose()),axis=0)
    #                 HSV_features = np.concatenate((HSV_features,HSV_feature.transpose()),axis=0)
    #
    #         RGB_features = torch.tensor(RGB_features).to(device)
    #         LAB_features = torch.tensor(LAB_features).to(device)
    #         HSV_features = torch.tensor(HSV_features).to(device)
    #
    #         # make similarity
    #         aff_sp_f = torch.nn.functional.cosine_similarity(sp_features_p1.unsqueeze(0), sp_features_p1.unsqueeze(1), dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p2.unsqueeze(0), sp_features_p2.unsqueeze(1),dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p3.unsqueeze(0), sp_features_p3.unsqueeze(1),dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p4.unsqueeze(0), sp_features_p4.unsqueeze(1),dim=-1)
    #
    #         aff_score = torch.nn.functional.cosine_similarity(sp_scores.unsqueeze(0), sp_scores.unsqueeze(1), dim=-1)
    #
    #         aff_color = torch.nn.functional.cosine_similarity(RGB_features.unsqueeze(0), RGB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(HSV_features.unsqueeze(0), HSV_features.unsqueeze(1), dim=-1)
    #
    #         aff_mat = aff_sp_f.to(device) + aff_color.to(device) + aff_score.to(device)
    #
    #         aff_mat_mask = aff_mat * (torch.tensor(adjacent_matrix, dtype=torch.float32) +
    #                                   torch.eye(adjacent_matrix.shape[0])).to(device)
    #
    #         trans_mat = aff_mat_mask / (torch.sum(aff_mat_mask, dim=0, keepdim=True)+1e-5)
    #
    #
    #         # 包括背景类 --> (21, sp_num)
    #         RW_cams = torch.tensor(sp_scores_normalize.t(), dtype=torch.float32)
    #         trans_mat = torch.tensor(trans_mat, dtype=torch.float32)
    #
    #         iter_TIME = 4
    #         alph = 1
    #
    #         for i in range(iter_TIME):
    #             if i == 0:
    #                 RW_cams = RW_cams
    #             else:
    #                 RW_cams = alph * torch.matmul(RW_cams, trans_mat)
    #             RW_cams = torch.nn.functional.normalize(RW_cams, p=1, dim=0)
    #
    #             if i+1 == iter_TIME:
    #                 RW_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_cams.dtype)
    #
    #                 for cls_idx in lables:
    #                     for idx, temp_label in enumerate(range(low_label, high_label + 1)):
    #                         sp_mask = sp_label == temp_label
    #                         sp_mask_tensor = torch.tensor(sp_mask)
    #                         # exclusive bg class
    #                         RW_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_cams[cls_idx + 1,idx] # single batch input
    #
    #                 RW_class_response_maps = F.upsample(RW_class_response_maps,size=(cam_H,cam_W),mode='bilinear', align_corners=True)
    #
    #         RW_class_response_maps = RW_class_response_maps.cpu().numpy()
    #         RW_class_response_maps = RW_class_response_maps - np.min(RW_class_response_maps,(-2,-1),keepdims = True)
    #         RW_class_response_maps = RW_class_response_maps / (1e-5 + np.max(RW_class_response_maps,(-2,-1),keepdims = True))
    #         RW_class_response_maps = torch.tensor(RW_class_response_maps).to(device)
    #
    #         mask_CAM = RW_class_response_maps * class_response_maps
    #         mask_CAM_valid_peak_list, mask_CAM_peak_score = self.get_peak_list(mask_CAM, lables, peak_threshold)
    #
    #         if len(mask_CAM_valid_peak_list) > 0:
    #             valid_peak_list = torch.stack(mask_CAM_valid_peak_list)
    #             peak_score = torch.tensor(mask_CAM_peak_score)
    #             return mask_CAM, valid_peak_list, peak_score
    #         else:
    #             return None

    # use proposal score to replace cam score v2
    # label assign
    def forward(self, input, lables, img, class_threshold=0, peak_threshold=10, retrieval_cfg=None, graph_op=None,
                resolution=7):
        assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
        with torch.no_grad():
            device = input.device

            # _, _, H, W = input.shape  # batchsize == 1
            H, W, _ = np.array(img).shape

            LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            HSV_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # classification network forwarding
            class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
            class_response_maps = class_response_maps.detach()
            org_size_class_response_maps = F.upsample(class_response_maps, size=(H, W), mode='bilinear',
                                                      align_corners=True)

            _, cls_num, cam_H, cam_W = class_response_maps.shape

            onehot_label = self.label2onehot(lables, cls_num)

            _, _, f_H_p1, f_W_p1 = p1.shape
            _, _, f_H_p2, f_W_p2 = p2.shape
            _, _, f_H_p3, f_W_p3 = p3.shape
            _, _, f_H_p4, f_W_p4 = p4.shape

            Align_op_p1 = RoIAlign((resolution, resolution), f_H_p1 / H, 0)
            Align_op_p2 = RoIAlign((resolution, resolution), f_H_p2 / H, 0)
            Align_op_p3 = RoIAlign((resolution, resolution), f_H_p3 / H, 0)
            Align_op_p4 = RoIAlign((resolution, resolution), f_H_p4 / H, 0)

            # graph operate
            sp_label = graph_op["sp_label"]
            adjacent_matrix = graph_op["adjacent_matrix"]
            img_id = graph_op["img_id"]
            belong_mat = graph_op["belong_mat"]
            proposal_score = graph_op["proposal_score"]

            # add bg_score
            bg_score = 1 - proposal_score.sum(-1)
            proposal_score = np.concatenate((bg_score[:, None], proposal_score), axis=1)

            # lzc edit 9-6
            max_idx = np.argmax(proposal_score, axis=-1)
            flag = np.zeros_like(max_idx)
            for label in lables:
                flag += (label + 1).item() == max_idx

            bg_score_const = np.zeros(proposal_score.shape[-1])
            bg_score_const[0] = 1
            proposal_score[flag == 0] = bg_score_const

            sp_scores = np.zeros((adjacent_matrix.shape[0], proposal_score.shape[-1]))  # sp_num, cls_num + 1
            for row_idx in range(belong_mat.shape[0]):
                score = proposal_score[row_idx]
                belong_row = belong_mat[row_idx] != 0

                sp_scores[belong_row] += score

            sp_scores = torch.tensor(sp_scores).to(device)
            sp_scores_normalize = torch.nn.functional.normalize(sp_scores, p=1, dim=-1)

            low_label = np.min(sp_label)
            high_label = np.max(sp_label)

            for idx, temp_label in enumerate(range(low_label, high_label + 1)):
                sp_mask = sp_label == temp_label

                # sp_mask_tensor = torch.tensor(sp_mask)

                ind_xy = np.nonzero(sp_mask)
                xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1

                mask = sp_mask[ymin:ymax, xmin:xmax]
                mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
                mask = torch.tensor(mask).to(device)

                box = torch.tensor([[0, xmin, ymin, xmax, ymax]], dtype=torch.float32).to(device)
                box_feature_p1 = Align_op_p1(p1, box)  # (1,7,7)
                box_feature_p2 = Align_op_p2(p2, box)  # (1,7,7)
                box_feature_p3 = Align_op_p3(p3, box)  # (1,7,7)
                box_feature_p4 = Align_op_p4(p4, box)  # (1,7,7)

                sp_feature_p1 = get_sp_f(box_feature_p1, mask, resolution)
                sp_feature_p2 = get_sp_f(box_feature_p2, mask, resolution)
                sp_feature_p3 = get_sp_f(box_feature_p3, mask, resolution)
                sp_feature_p4 = get_sp_f(box_feature_p4, mask, resolution)

                if idx == 0:
                    sp_features_p1 = sp_feature_p1
                    sp_features_p2 = sp_feature_p2
                    sp_features_p3 = sp_feature_p3
                    sp_features_p4 = sp_feature_p4

                    RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
                    LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
                    HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
                    for color_idx in range(1, 3):
                        temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
                        temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
                        temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])

                        RGB_feature = np.concatenate((RGB_feature, temp_feature_rgb), axis=0)
                        LAB_feature = np.concatenate((LAB_feature, temp_feature_lab), axis=0)
                        HSV_feature = np.concatenate((HSV_feature, temp_feature_hsv), axis=0)

                    RGB_features = RGB_feature.transpose()
                    LAB_features = LAB_feature.transpose()
                    HSV_features = HSV_feature.transpose()

                else:
                    sp_features_p1 = torch.cat((sp_features_p1, sp_feature_p1), dim=0)
                    sp_features_p2 = torch.cat((sp_features_p2, sp_feature_p2), dim=0)
                    sp_features_p3 = torch.cat((sp_features_p3, sp_feature_p3), dim=0)
                    sp_features_p4 = torch.cat((sp_features_p4, sp_feature_p4), dim=0)

                    RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
                    LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
                    HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
                    for color_idx in range(1, 3):
                        temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
                        temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
                        temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])

                        RGB_feature = np.concatenate((RGB_feature, temp_feature_rgb), axis=0)
                        LAB_feature = np.concatenate((LAB_feature, temp_feature_lab), axis=0)
                        HSV_feature = np.concatenate((HSV_feature, temp_feature_hsv), axis=0)

                    RGB_features = np.concatenate((RGB_features, RGB_feature.transpose()), axis=0)
                    LAB_features = np.concatenate((LAB_features, LAB_feature.transpose()), axis=0)
                    HSV_features = np.concatenate((HSV_features, HSV_feature.transpose()), axis=0)

            RGB_features = torch.tensor(RGB_features).to(device)
            LAB_features = torch.tensor(LAB_features).to(device)
            HSV_features = torch.tensor(HSV_features).to(device)

            # make similarity
            aff_sp_f = torch.nn.functional.cosine_similarity(sp_features_p1.unsqueeze(0), sp_features_p1.unsqueeze(1),
                                                             dim=-1) * \
                       torch.nn.functional.cosine_similarity(sp_features_p2.unsqueeze(0), sp_features_p2.unsqueeze(1),
                                                             dim=-1) * \
                       torch.nn.functional.cosine_similarity(sp_features_p3.unsqueeze(0), sp_features_p3.unsqueeze(1),
                                                             dim=-1) * \
                       torch.nn.functional.cosine_similarity(sp_features_p4.unsqueeze(0), sp_features_p4.unsqueeze(1),
                                                             dim=-1)

            aff_score = torch.nn.functional.cosine_similarity(sp_scores.unsqueeze(0), sp_scores.unsqueeze(1), dim=-1)

            aff_color = torch.nn.functional.cosine_similarity(RGB_features.unsqueeze(0), RGB_features.unsqueeze(1),
                                                              dim=-1) * \
                        torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1),
                                                              dim=-1) * \
                        torch.nn.functional.cosine_similarity(HSV_features.unsqueeze(0), HSV_features.unsqueeze(1),
                                                              dim=-1)

            aff_mat = aff_sp_f.to(device) + aff_color.to(device) + aff_score.to(device)

            aff_mat_mask = aff_mat * (torch.tensor(adjacent_matrix, dtype=torch.float32) +
                                      torch.eye(adjacent_matrix.shape[0])).to(device)

            trans_mat = aff_mat_mask / (torch.sum(aff_mat_mask, dim=0, keepdim=True) + 1e-5)

            # 包括背景类 --> (21, sp_num)
            RW_cams = torch.tensor(sp_scores_normalize.t(), dtype=torch.float32)
            trans_mat = torch.tensor(trans_mat, dtype=torch.float32)

            iter_TIME = 4

            for i in range(iter_TIME):
                if i == 0:
                    RW_cams = RW_cams
                else:
                    RW_cams = torch.matmul(RW_cams, trans_mat)
                RW_cams = torch.nn.functional.normalize(RW_cams, p=1, dim=0)

                if i + 1 == iter_TIME:
                    RW_class_response_maps = torch.zeros_like(org_size_class_response_maps, dtype=RW_cams.dtype)

                    for cls_idx in lables:
                        for idx, temp_label in enumerate(range(low_label, high_label + 1)):
                            sp_mask = sp_label == temp_label
                            sp_mask_tensor = torch.tensor(sp_mask)
                            # exclusive bg class
                            RW_class_response_maps[0, cls_idx, sp_mask_tensor] = RW_cams[
                                cls_idx + 1, idx]  # single batch input

                    RW_class_response_maps = F.upsample(RW_class_response_maps, size=(cam_H, cam_W), mode='bilinear',
                                                        align_corners=True)

            RW_class_response_maps = RW_class_response_maps.cpu().numpy()
            RW_class_response_maps = RW_class_response_maps - np.min(RW_class_response_maps, (-2, -1), keepdims=True)
            RW_class_response_maps = RW_class_response_maps / (
                        1e-5 + np.max(RW_class_response_maps, (-2, -1), keepdims=True))
            RW_class_response_maps = torch.tensor(RW_class_response_maps).to(device)

            # RW_valid_peak_list, RW_peak_score = self.get_peak_list(RW_class_response_maps,lables,0.5)

            mask_CAM = RW_class_response_maps * class_response_maps
            mask_CAM_valid_peak_list, mask_CAM_peak_score = self.get_peak_list(mask_CAM, lables, peak_threshold)

            if len(mask_CAM_valid_peak_list) > 0:
                valid_peak_list = torch.stack(mask_CAM_valid_peak_list)
                peak_score = torch.tensor(mask_CAM_peak_score)
                return RW_class_response_maps, valid_peak_list, peak_score
            else:
                return None

    # # edit 9.4 lzc
    # def forward(self, input, lables, img,class_threshold=0, peak_threshold=10, retrieval_cfg=None,graph_op=None,resolution=7):
    #     assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
    #     dir = "./voc_visual"
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #     with torch.no_grad():
    #
    #         # _, _, H, W = input.shape  # batchsize == 1
    #         H,W,_ = np.array(img).shape
    #
    #         LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #         HSV_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #
    #         # classification network forwarding
    #         class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
    #
    #         _, cls_num, cam_H, cam_W = class_response_maps.shape
    #
    #         onehot_label = torch.tensor(self.label2onehot(lables,cls_num))
    #         class_response_maps = class_response_maps * torch.tensor(onehot_label).view(1,cls_num,1,1)
    #
    #         class_response_maps = class_response_maps.detach()
    #         org_size_class_response_maps = F.upsample(class_response_maps, size=(H,W), mode='bilinear', align_corners=True)
    #
    #         # use p2 as feature
    #         feature = p4
    #
    #         _, _, f_H, f_W = feature.shape
    #
    #         # spatial_scale
    #         spatial_scale = f_H / H
    #         Align_op = RoIAlign((resolution, resolution), spatial_scale, 0)
    #
    #         # graph operate
    #         sp_label = graph_op["sp_label"]
    #         adjacent_matrix = graph_op["adjacent_matrix"]
    #         img_id = graph_op["img_id"]
    #         belong_mat = graph_op["belong_mat"]
    #         proposal_score = graph_op["proposal_score"]
    #         print(img_id)
    #
    #         # add bg_score
    #         bg_score = 1 - proposal_score.sum(-1)
    #         proposal_score = np.concatenate((bg_score[:,None],proposal_score),axis=1)
    #
    #         sp_scores = np.zeros((adjacent_matrix.shape[0],proposal_score.shape[-1])) # sp_num, cls_num + 1
    #         for row_idx in range(belong_mat.shape[0]):
    #             score = proposal_score[row_idx]
    #             belong_row = belong_mat[row_idx] != 0
    #
    #             sp_scores[belong_row] += score
    #
    #         sp_scores = torch.tensor(sp_scores)
    #         # sp_scores_normalize = torch.nn.functional.normalize(sp_scores,p=1,dim=-1)
    #
    #         low_label = np.min(sp_label)
    #         high_label = np.max(sp_label)
    #
    #         for idx, temp_label in tqdm(enumerate(range(low_label, high_label + 1)),total=adjacent_matrix.shape[0]):
    #             sp_mask = sp_label == temp_label
    #
    #             sp_mask_tensor = torch.tensor(sp_mask)
    #             cam_scores = (sp_mask_tensor * org_size_class_response_maps).sum((-2,-1)) / sp_mask_tensor.sum() # --> (1, cls_num)
    #
    #             ind_xy = np.nonzero(sp_mask)
    #             xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
    #
    #             mask = sp_mask[ymin:ymax, xmin:xmax]
    #             mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
    #             mask = torch.tensor(mask)
    #
    #             box = torch.tensor([[0, xmin, ymin, xmax, ymax]],dtype=torch.float32)
    #             box_feature = Align_op(feature, box)  # (1,7,7)
    #
    #             sp_f = (mask * box_feature).view(-1, resolution, resolution)  # (1,dim,7,7) --> (dim,7,7)
    #             sp_f = sp_f.sum((-2,-1))  # (dim)
    #             sp_f = sp_f / (mask.sum()+1e-5)
    #
    #             sp_f = sp_f[None, :]
    #
    #             if idx == 0:
    #                 sp_cams = cam_scores
    #                 sp_features = sp_f
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1,3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature,temp_feature_rgb),axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature,temp_feature_lab),axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature,temp_feature_hsv),axis=0)
    #
    #                 RGB_features = RGB_feature.transpose()
    #                 LAB_features = LAB_feature.transpose()
    #                 HSV_features = HSV_feature.transpose()
    #
    #             else:
    #                 sp_cams = torch.cat((sp_cams, cam_scores), dim=0)
    #                 sp_features = torch.cat((sp_features, sp_f), dim=0)
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1, 3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature, temp_feature_rgb), axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature, temp_feature_lab), axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature, temp_feature_hsv), axis=0)
    #
    #                 RGB_features = np.concatenate((RGB_features,RGB_feature.transpose()),axis=0)
    #                 LAB_features = np.concatenate((LAB_features,LAB_feature.transpose()),axis=0)
    #                 HSV_features = np.concatenate((HSV_features,HSV_feature.transpose()),axis=0)
    #
    #         RGB_features = torch.tensor(RGB_features)
    #         LAB_features = torch.tensor(LAB_features)
    #         HSV_features = torch.tensor(HSV_features)
    #
    #
    #         # make similarity
    #         aff_sp_f = torch.nn.functional.cosine_similarity(sp_features.unsqueeze(0), sp_features.unsqueeze(1), dim=-1)
    #         aff_score = torch.nn.functional.cosine_similarity(sp_scores.unsqueeze(0), sp_scores.unsqueeze(1), dim=-1)
    #         aff_color = torch.nn.functional.cosine_similarity(RGB_features.unsqueeze(0), RGB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(HSV_features.unsqueeze(0), HSV_features.unsqueeze(1), dim=-1)
    #
    #
    #         aff_mat = aff_sp_f + aff_score + aff_color
    #         # aff_mat = aff_sp_f * aff_score * aff_rgb
    #
    #         aff_mat_mask = aff_mat * (torch.tensor(adjacent_matrix, dtype=torch.float32) + torch.eye(adjacent_matrix.shape[0]))
    #
    #         trans_mat = aff_mat_mask / (torch.sum(aff_mat_mask, dim=0, keepdim=True)+1e-5)
    #
    #         # logt = 4
    #         #
    #         # for _ in range(logt):
    #         #     trans_mat = torch.matmul(trans_mat, trans_mat)
    #
    #         # trans_mat = torch.linalg.inv(1e-5 * torch.eye(aff_mat.shape[0]) + torch.tensor(aff_mat_mask,dtype=torch.float32)) @ torch.tensor(aff_mat_mask,dtype=torch.float32)
    #
    #         plt.close()
    #         fig,axx = plt.subplots(2, 3)
    #
    #         sns.heatmap(aff_sp_f.numpy(),cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,0])
    #         axx[0,0].set_title("aff_sp_f")
    #
    #         sns.heatmap(aff_score.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,1])
    #         axx[0,1].set_title("aff_score")
    #
    #         # axx[0,2].axis("off")
    #         sns.heatmap(aff_color.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,2])
    #         axx[0,2].set_title("aff_color")
    #
    #         sns.heatmap(aff_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,0])
    #         axx[1,0].set_title("aff_mat")
    #
    #         sns.heatmap(aff_mat_mask.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,1])
    #         axx[1,1].set_title("aff_mat_mask")
    #
    #         sns.heatmap(trans_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,2])
    #         axx[1,2].set_title("trans_mat")
    #
    #         print(os.path.join(dir,str(img_id)+"_aff.jpg"))
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+"_aff.jpg"), dpi=300)
    #
    #         RW_cams = torch.tensor(sp_cams.t(), dtype=torch.float32)
    #         trans_mat = torch.tensor(trans_mat, dtype=torch.float32)
    #
    #         cam_valid_peak_list, cam_peak_score = self.get_peak_list(class_response_maps,lables,peak_threshold)
    #         response_maps = [class_response_maps]
    #         valid_peak_list = [cam_valid_peak_list]
    #         scale = 0.7
    #
    #         # RW_cams = torch.matmul(torch.tensor(trans_mat.t(),dtype=torch.float32),torch.tensor(sp_cams,dtype=torch.float32))
    #         # RW_cams = RW_cams.t()
    #
    #         iter_TIME = 4
    #         alph = 0.7
    #
    #         for _ in range(iter_TIME):
    #             RW_cams = alph * torch.matmul(RW_cams, trans_mat) + (1 - alph) * torch.tensor(sp_cams.t(),dtype=torch.float32)
    #             RW_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_cams.dtype)
    #
    #             for cls_idx in lables:
    #                 for idx, temp_label in enumerate(range(low_label, high_label + 1)):
    #                     sp_mask = sp_label == temp_label
    #                     sp_mask_tensor = torch.tensor(sp_mask)
    #                     RW_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_cams[cls_idx,temp_label] # single batch input
    #
    #             RW_class_response_maps = F.upsample(RW_class_response_maps,size=(cam_H,cam_W),mode='bilinear', align_corners=True)
    #
    #             RW_valid_peak_list, RW_peak_score = self.get_peak_list(RW_class_response_maps,lables,peak_threshold)
    #
    #             response_maps.append(RW_class_response_maps)
    #             valid_peak_list.append(RW_valid_peak_list)
    #
    #         cam_num = onehot_label.sum().item()
    #         plt.close()
    #         f, axarr = plt.subplots(len(response_maps), int(cam_num + 1))
    #         plt.tight_layout()
    #
    #         for idx in range(len(response_maps)):
    #             if idx == 0:
    #                 axarr[idx, 0].imshow(img)
    #             else:
    #                 axarr[idx, 0].imshow(mark_boundaries(img, sp_label))
    #
    #             axarr[idx, 0].axis("off")
    #
    #             for i,cls_idx in enumerate(lables):
    #                 CRM = response_maps[idx]
    #
    #                 cam = CRM[0, cls_idx].detach().cpu().numpy()
    #                 cam = cam - np.min(cam)
    #                 cam = cam / np.max(cam)
    #                 cam = np.uint8(255 * cam)
    #
    #                 heatmap = cv2.applyColorMap(cv2.resize(cam, (W, H)), cv2.COLORMAP_JET)
    #                 heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #                 result = heatmap * scale + (img) * (1 - scale)
    #                 result = np.uint8(result)
    #                 result = cv2.resize(result, (int(cam_W*self.sub_pixel_locating_factor),int(cam_H*self.sub_pixel_locating_factor)))
    #
    #                 axarr[idx, i+1].imshow(result)
    #                 axarr[idx, i+1].axis("off")
    #                 axarr[idx, i+1].set_title(voc_id_name_map[cls_idx.item()+1])
    #
    #                 VPL = valid_peak_list[idx]
    #
    #                 for peak in VPL:
    #                     if peak[1] == cls_idx:
    #                         axarr[idx, i+1].scatter(peak[3],peak[2], color='purple',marker='*',edgecolors='purple', s=17)
    #
    #                 axarr[idx, i+1].axis("off")
    #
    #         # plt.show()
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+".jpg"), dpi=300)
    #
    #         # if len(peak_score) > 0:
    #         #     valid_peak_list = torch.stack(valid_peak_list)
    #         #     peak_score = torch.tensor(peak_score)
    #         #     # classification confidence scores, class-aware and instance-aware visual cues
    #         #     return class_response_maps, valid_peak_list, peak_score
    #         # else:
    #         #     return None

    # def forward(self, input, lables, img, class_threshold=0, peak_threshold=10, retrieval_cfg=None, graph_op=None,resolution=7):
    #
    #     assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
    #     with torch.no_grad():
    #
    #         _, _, H, W = input.shape  # batchsize == 1
    #
    #         # classification network forwarding
    #         class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
    #
    #         _, cls_num, cam_H, cam_W = class_response_maps.shape
    #
    #         onehot_label = torch.tensor(self.label2onehot(lables,cls_num))
    #         class_response_maps = class_response_maps * torch.tensor(onehot_label).view(1,cls_num,1,1)
    #
    #         # gap_cam = torch.nn.AdaptiveMaxPool2d((1,1))(class_response_maps)
    #
    #         class_response_maps = class_response_maps.detach()
    #         orig_img_size = img.shape[:2]
    #
    #         class_response_maps_normalize = F.upsample(class_response_maps, orig_img_size, mode='bilinear',align_corners=False)
    #         class_response_maps_normalize = class_response_maps_normalize.numpy()[0]
    #         class_response_maps_normalize = class_response_maps_normalize - np.min(class_response_maps_normalize,axis=(-2,-1))[:, None,None]
    #         class_response_maps_normalize = class_response_maps_normalize / np.max(1e-5 + class_response_maps_normalize,axis=(-2,-1))[:,None, None]
    #         bg_score = np.power(1 - np.max(class_response_maps_normalize, axis=0, keepdims=True), 1)
    #         bgcam_score = np.concatenate((bg_score, class_response_maps_normalize[lables].reshape(len(lables),
    #                                                                                               class_response_maps_normalize.shape[1],
    #                                                                                               class_response_maps_normalize.shape[2])), axis=0)
    #         crf_score = crf_inference(img, bgcam_score, t=1, labels=bgcam_score.shape[0])
    #         crf_score = torch.tensor(crf_score)[None, 1:]
    #         crf_score = F.upsample(crf_score, size=(cam_H, cam_W), mode='bilinear', align_corners=True)
    #
    #         cam_valid_peak_list, cam_peak_score = self.get_peak_list(class_response_maps,lables,peak_threshold)
    #         crf_valid_peak_list, crf_peak_score = self.get_peak_list(crf_score,torch.arange(len(lables)),0.5)
    #
    #         cam_num = onehot_label.sum().item()
    #         f, axarr = plt.subplots(2, int(cam_num + 1))
    #         plt.tight_layout()
    #
    #         response_maps = [class_response_maps,crf_score]
    #         valid_peak_list = [cam_valid_peak_list,crf_valid_peak_list]
    #         scale = 0.7
    #         img = cv2.resize(img,(448,448))
    #
    #         img_id = graph_op["img_id"]
    #
    #         # axarr[0, 0].set_title(str(torch.nn.functional.sigmoid(gap_cam).flatten().numpy()[lables]))
    #
    #         for idx in range(2):
    #             axarr[idx, 0].imshow(img)
    #             axarr[idx, 0].axis("off")
    #
    #             for i,cls_idx in enumerate(lables):
    #                 CRM = response_maps[idx]
    #                 if idx == 0:
    #                     cam = CRM[0, cls_idx].detach().cpu().numpy()
    #                 else:
    #                     cam = CRM[0, i].detach().cpu().numpy()
    #
    #                 cam = cam - np.min(cam)
    #                 cam = cam / np.max(cam)
    #                 cam = np.uint8(255 * cam)
    #
    #                 heatmap = cv2.applyColorMap(cv2.resize(cam, (W, H)), cv2.COLORMAP_JET)
    #                 heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #                 result = heatmap * scale + (img) * (1 - scale)
    #                 result = np.uint8(result)
    #                 result = cv2.resize(result, (int(cam_W*self.sub_pixel_locating_factor),int(cam_H*self.sub_pixel_locating_factor)))
    #
    #                 axarr[idx, i+1].imshow(result)
    #                 axarr[idx, i+1].axis("off")
    #                 axarr[idx, i+1].set_title(voc_id_name_map[cls_idx.item()+1])
    #
    #                 VPL = valid_peak_list[idx]
    #
    #                 for peak in VPL:
    #                     if idx == 0:
    #                         if peak[1] == cls_idx:
    #                             axarr[idx, i+1].scatter(peak[3],peak[2], color='purple',marker='*',edgecolors='purple', s=17)
    #                     else:
    #                         if peak[1] == i:
    #                             axarr[idx, i + 1].scatter(peak[3], peak[2], color='purple', marker='*',
    #                                                       edgecolors='purple', s=17)
    #
    #                 axarr[idx, i+1].axis("off")
    #
    #         dir = "./voc_visual"
    #         if not os.path.exists(dir):
    #             os.makedirs(dir)
    #         plt.savefig(os.path.join(dir,str(img_id)+".jpg"), dpi=300)

    # # cam score and proposal score
    # def forward(self, input, lables, img, class_threshold=0, peak_threshold=10, retrieval_cfg=None, graph_op=None,resolution=7):
    #     assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
    #     dir = "./voc_visual"
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #     with torch.no_grad():
    #
    #         # _, _, H, W = input.shape  # batchsize == 1
    #         H,W,_ = np.array(img).shape
    #
    #         LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #         HSV_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #
    #         # classification network forwarding
    #         class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
    #         class_response_maps = class_response_maps.detach()
    #         org_size_class_response_maps = F.upsample(class_response_maps, size=(H,W), mode='bilinear', align_corners=True)
    #
    #         _, cls_num, cam_H, cam_W = class_response_maps.shape
    #
    #         onehot_label = torch.tensor(self.label2onehot(lables,cls_num))
    #
    #         _, _, f_H_p1, f_W_p1 = p1.shape
    #         _, _, f_H_p2, f_W_p2 = p2.shape
    #         _, _, f_H_p3, f_W_p3 = p3.shape
    #         _, _, f_H_p4, f_W_p4 = p4.shape
    #
    #         Align_op_p1 = RoIAlign((resolution, resolution), f_H_p1/H, 0)
    #         Align_op_p2 = RoIAlign((resolution, resolution), f_H_p2/H, 0)
    #         Align_op_p3 = RoIAlign((resolution, resolution), f_H_p3/H, 0)
    #         Align_op_p4 = RoIAlign((resolution, resolution), f_H_p4/H, 0)
    #
    #         # graph operate
    #         sp_label = graph_op["sp_label"]
    #         adjacent_matrix = graph_op["adjacent_matrix"]
    #         img_id = graph_op["img_id"]
    #         belong_mat = graph_op["belong_mat"]
    #         proposal_score = graph_op["proposal_score"]
    #         print(img_id)
    #
    #         # add bg_score
    #         bg_score = 1 - proposal_score.sum(-1)
    #         proposal_score = np.concatenate((bg_score[:,None],proposal_score),axis=1)
    #
    #         sp_scores = np.zeros((adjacent_matrix.shape[0],proposal_score.shape[-1])) # sp_num, cls_num + 1
    #         for row_idx in range(belong_mat.shape[0]):
    #             score = proposal_score[row_idx]
    #             belong_row = belong_mat[row_idx] != 0
    #
    #             sp_scores[belong_row] += score
    #
    #         sp_scores = torch.tensor(sp_scores)
    #         sp_scores_normalize = torch.nn.functional.normalize(sp_scores,p=1,dim=-1)
    #
    #         low_label = np.min(sp_label)
    #         high_label = np.max(sp_label)
    #
    #         for idx, temp_label in tqdm(enumerate(range(low_label, high_label + 1)),total=adjacent_matrix.shape[0]):
    #             sp_mask = sp_label == temp_label
    #             if idx == 0:
    #                 sp_proposal = sp_mask[None,:,:]
    #             else:
    #                 sp_proposal = np.concatenate((sp_proposal, sp_mask[None,:,:]),axis=0)
    #
    #             sp_mask_tensor = torch.tensor(sp_mask)
    #             cam_scores = (sp_mask_tensor * org_size_class_response_maps).sum(-1).sum(-1) / sp_mask_tensor.sum() # --> (1, cls_num)
    #
    #             ind_xy = np.nonzero(sp_mask)
    #             xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
    #
    #             mask = sp_mask[ymin:ymax, xmin:xmax]
    #             mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
    #             mask = torch.tensor(mask)
    #
    #             box = torch.tensor([[0, xmin, ymin, xmax, ymax]],dtype=torch.float32)
    #             box_feature_p1 = Align_op_p1(p1, box)  # (1,7,7)
    #             box_feature_p2 = Align_op_p2(p2, box)  # (1,7,7)
    #             box_feature_p3 = Align_op_p3(p3, box)  # (1,7,7)
    #             box_feature_p4 = Align_op_p4(p4, box)  # (1,7,7)
    #
    #             sp_feature_p1 = get_sp_f(box_feature_p1,mask,resolution)
    #             sp_feature_p2 = get_sp_f(box_feature_p2,mask,resolution)
    #             sp_feature_p3 = get_sp_f(box_feature_p3,mask,resolution)
    #             sp_feature_p4 = get_sp_f(box_feature_p4,mask,resolution)
    #
    #             if idx == 0:
    #                 sp_cams = cam_scores
    #             else:
    #                 sp_cams = torch.cat((sp_cams, cam_scores), dim=0)
    #
    #             if idx == 0:
    #                 sp_features_p1 = sp_feature_p1
    #                 sp_features_p2 = sp_feature_p2
    #                 sp_features_p3 = sp_feature_p3
    #                 sp_features_p4 = sp_feature_p4
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1,3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature,temp_feature_rgb),axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature,temp_feature_lab),axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature,temp_feature_hsv),axis=0)
    #
    #                 RGB_features = RGB_feature.transpose()
    #                 LAB_features = LAB_feature.transpose()
    #                 HSV_features = HSV_feature.transpose()
    #
    #             else:
    #                 sp_features_p1 = torch.cat((sp_features_p1, sp_feature_p1),dim=0)
    #                 sp_features_p2 = torch.cat((sp_features_p2, sp_feature_p2),dim=0)
    #                 sp_features_p3 = torch.cat((sp_features_p3, sp_feature_p3),dim=0)
    #                 sp_features_p4 = torch.cat((sp_features_p4, sp_feature_p4),dim=0)
    #
    #                 RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
    #                 for color_idx in range(1, 3):
    #                     temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #                     temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
    #
    #                     RGB_feature = np.concatenate((RGB_feature, temp_feature_rgb), axis=0)
    #                     LAB_feature = np.concatenate((LAB_feature, temp_feature_lab), axis=0)
    #                     HSV_feature = np.concatenate((HSV_feature, temp_feature_hsv), axis=0)
    #
    #                 RGB_features = np.concatenate((RGB_features,RGB_feature.transpose()),axis=0)
    #                 LAB_features = np.concatenate((LAB_features,LAB_feature.transpose()),axis=0)
    #                 HSV_features = np.concatenate((HSV_features,HSV_feature.transpose()),axis=0)
    #
    #         RGB_features = torch.tensor(RGB_features)
    #         LAB_features = torch.tensor(LAB_features)
    #         HSV_features = torch.tensor(HSV_features)
    #
    #
    #         # make similarity
    #         aff_sp_f = torch.nn.functional.cosine_similarity(sp_features_p1.unsqueeze(0), sp_features_p1.unsqueeze(1), dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p2.unsqueeze(0), sp_features_p2.unsqueeze(1),dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p3.unsqueeze(0), sp_features_p3.unsqueeze(1),dim=-1) * \
    #                    torch.nn.functional.cosine_similarity(sp_features_p4.unsqueeze(0), sp_features_p4.unsqueeze(1),dim=-1)
    #
    #         aff_score = torch.nn.functional.cosine_similarity(sp_scores.unsqueeze(0), sp_scores.unsqueeze(1), dim=-1)
    #
    #         aff_color = torch.nn.functional.cosine_similarity(RGB_features.unsqueeze(0), RGB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1), dim=-1) * \
    #                     torch.nn.functional.cosine_similarity(HSV_features.unsqueeze(0), HSV_features.unsqueeze(1), dim=-1)
    #
    #
    #         aff_mat = aff_sp_f + aff_color + aff_score
    #
    #         aff_mat_mask = aff_mat * (torch.tensor(adjacent_matrix, dtype=torch.float32) + torch.eye(adjacent_matrix.shape[0]))
    #
    #         trans_mat = aff_mat_mask / (torch.sum(aff_mat_mask, dim=0, keepdim=True)+1e-5)
    #
    #         plt.close()
    #         fig,axx = plt.subplots(2, 3)
    #
    #         sns.heatmap(aff_sp_f.numpy(),cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,0])
    #         axx[0,0].set_title("aff_sp_f")
    #
    #         sns.heatmap(aff_score.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,1])
    #         axx[0,1].set_title("aff_score")
    #
    #         # axx[0,2].axis("off")
    #         sns.heatmap(aff_color.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,2])
    #         axx[0,2].set_title("aff_color")
    #
    #         sns.heatmap(aff_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,0])
    #         axx[1,0].set_title("aff_mat")
    #
    #         sns.heatmap(aff_mat_mask.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,1])
    #         axx[1,1].set_title("aff_mat_mask")
    #
    #         sns.heatmap(trans_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,2])
    #         axx[1,2].set_title("trans_mat")
    #
    #         print(os.path.join(dir,str(img_id)+"_aff.jpg"))
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+"_aff.jpg"), dpi=300)
    #
    #         # 包括背景类 --> (21, sp_num)
    #         # RW_cams = torch.tensor(sp_scores_normalize.t(), dtype=torch.float32)
    #         RW_sp_cams =  torch.tensor(sp_cams.t(), dtype=torch.float32)
    #         RW_sp_normalize_cams = torch.tensor(sp_scores_normalize.t(), dtype=torch.float32)
    #
    #         trans_mat = torch.tensor(trans_mat, dtype=torch.float32)
    #
    #         cam_valid_peak_list, cam_peak_score = self.get_peak_list(class_response_maps,lables,peak_threshold)
    #         response_maps = [class_response_maps]
    #         valid_peak_list = [cam_valid_peak_list]
    #         scale = 0.7
    #
    #         iter_TIME = 4 + 1
    #
    #         for i in range(iter_TIME):
    #             if i == 0:
    #                 RW_sp_cams = RW_sp_cams
    #                 RW_sp_normalize_cams = RW_sp_normalize_cams
    #             else:
    #                 RW_sp_cams = torch.matmul(RW_sp_cams, trans_mat)
    #                 RW_sp_normalize_cams = torch.matmul(RW_sp_normalize_cams, trans_mat)
    #
    #             RW_sp_normalize_cams = torch.nn.functional.normalize(RW_sp_normalize_cams, p=1, dim=0)
    #
    #             RW_sp_cams_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_sp_cams.dtype)
    #             RW_sp_normalize_cams_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_sp_normalize_cams.dtype)
    #
    #             for cls_idx in lables:
    #                 for idx, temp_label in enumerate(range(low_label, high_label + 1)):
    #                     sp_mask = sp_label == temp_label
    #                     sp_mask_tensor = torch.tensor(sp_mask)
    #                     # exclusive bg class
    #                     RW_sp_cams_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_sp_cams[cls_idx,idx] # single batch input
    #                     RW_sp_normalize_cams_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_sp_normalize_cams[cls_idx + 1,idx] # single batch input
    #
    #             RW_class_response_maps = RW_sp_cams_class_response_maps * RW_sp_normalize_cams_class_response_maps
    #
    #             RW_class_response_maps = F.upsample(RW_class_response_maps,size=(cam_H,cam_W),mode='bilinear', align_corners=True)
    #
    #             RW_valid_peak_list, RW_peak_score = self.get_peak_list(RW_class_response_maps,lables,0,self._lzc_filter)
    #
    #             response_maps.append(RW_class_response_maps)
    #             valid_peak_list.append(RW_valid_peak_list)
    #
    #         RW_class_response_maps = RW_class_response_maps.numpy()
    #         RW_class_response_maps = RW_class_response_maps - np.min(RW_class_response_maps,(-2,-1),keepdims = True)
    #         RW_class_response_maps = RW_class_response_maps / np.max(RW_class_response_maps,(-2,-1),keepdims = True)
    #         RW_class_response_maps = torch.tensor(RW_class_response_maps)
    #
    #         print(RW_class_response_maps)
    #         mask_CAM = RW_class_response_maps * class_response_maps
    #         mask_CAM_valid_peak_list, mask_CAM_peak_score = self.get_peak_list(mask_CAM, lables, peak_threshold)
    #         response_maps.append(mask_CAM)
    #         valid_peak_list.append(mask_CAM_valid_peak_list)
    #
    #         cam_num = onehot_label.sum().item()
    #         plt.close()
    #         f, axarr = plt.subplots(len(response_maps), int(cam_num + 1))
    #         plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.1, hspace=0.1)
    #
    #         for idx in range(len(response_maps)):
    #             if idx == 0:
    #                 axarr[idx, 0].imshow(img)
    #             else:
    #                 axarr[idx, 0].imshow(mark_boundaries(img, sp_label))
    #
    #             axarr[idx, 0].axis("off")
    #
    #             for i,cls_idx in enumerate(lables):
    #                 CRM = response_maps[idx]
    #
    #                 cam = CRM[0, cls_idx].detach().cpu().numpy()
    #                 cam = cam - np.min(cam)
    #                 cam = cam / np.max(cam)
    #                 cam = np.uint8(255 * cam)
    #
    #                 heatmap = cv2.applyColorMap(cv2.resize(cam, (W, H)), cv2.COLORMAP_JET)
    #                 heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #                 result = heatmap * scale + (img) * (1 - scale)
    #                 result = np.uint8(result)
    #                 result = cv2.resize(result, (int(cam_W*self.sub_pixel_locating_factor),int(cam_H*self.sub_pixel_locating_factor)))
    #
    #                 axarr[idx, i+1].imshow(result)
    #                 axarr[idx, i+1].axis("off")
    #                 axarr[idx, i+1].set_title(voc_id_name_map[cls_idx.item()+1])
    #
    #                 VPL = valid_peak_list[idx]
    #
    #                 for peak in VPL:
    #                     if peak[1] == cls_idx:
    #                         axarr[idx, i+1].scatter(peak[3],peak[2], color='purple',marker='*',edgecolors='purple', s=17)
    #
    #                 axarr[idx, i+1].axis("off")
    #
    #         # plt.show()
    #
    #         plt.savefig(os.path.join(dir,str(img_id)+".jpg"), dpi=300)
    #
    #         # if len(peak_score) > 0:
    #         #     valid_peak_list = torch.stack(valid_peak_list)
    #         #     peak_score = torch.tensor(peak_score)
    #         #     # classification confidence scores, class-aware and instance-aware visual cues
    #         #     return class_response_maps, valid_peak_list, peak_score
    #         # else:
    #         #     return None

    # def forward(self, input, lables,img,class_threshold=0, peak_threshold=10, retrieval_cfg=None,graph_op=None,resolution=7):
    #     assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
    #     with torch.no_grad():
    #         device = input.device
    #         _, _, H, W = input.shape  # batchsize == 1
    #
    #         # classification network forwarding
    #         class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff, self).forward(input)
    #
    #
    #         _, cls_num, cam_H, cam_W = class_response_maps.shape
    #
    #         onehot_label = torch.tensor(self.label2onehot(lables,cls_num)).to(device)
    #         class_response_maps = class_response_maps * torch.tensor(onehot_label).view(1,cls_num,1,1)
    #
    #         org_H,org_W,_ = img.shape
    #         org_size_class_response_maps = F.upsample(class_response_maps, size=(org_H,org_W), mode='bilinear', align_corners=True)
    #
    #         # use p2 as feature
    #         feature = p2
    #
    #         _, _, f_H, f_W = feature.shape
    #
    #         # spatial_scale
    #         spatial_scale = f_H / H
    #         Align_op = RoIAlign((resolution, resolution), spatial_scale, 0)
    #
    #         # graph operate
    #         sp_label = graph_op["sp_label"]
    #         adjacent_matrix = graph_op["adjacent_matrix"]
    #
    #         sp_num = adjacent_matrix.shape[0]
    #
    #         for idx, temp_label in enumerate(range(sp_num)):
    #             sp_mask = sp_label == temp_label
    #
    #             sp_mask_tensor = torch.tensor(sp_mask).to(device)
    #             cam_scores = (sp_mask_tensor * org_size_class_response_maps).sum(-1).sum(-1) / sp_mask_tensor.sum() # --> (1, cls_num)
    #
    #             ind_xy = np.nonzero(sp_mask)
    #             xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
    #
    #             mask = sp_mask[ymin:ymax, xmin:xmax]
    #             mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
    #             mask = torch.tensor(mask).to(device)
    #
    #             box = torch.tensor([[0, xmin, ymin, xmax, ymax]],dtype=torch.float32).to(device)
    #             box_feature = Align_op(feature, box)  # (1,7,7)
    #
    #             sp_f = (mask * box_feature).view(-1, resolution, resolution)  # (1,dim,7,7) --> (dim,7,7)
    #             sp_f = sp_f.view(-1, resolution * resolution).sum(-1)  # (dim)
    #             sp_f = sp_f / mask.sum()
    #
    #             sp_f = sp_f[None, :]
    #
    #             if idx == 0:
    #                 sp_cams = cam_scores
    #                 sp_features = sp_f
    #             else:
    #                 sp_cams = torch.cat((sp_cams, cam_scores), dim=0)
    #                 sp_features = torch.cat((sp_features, sp_f), dim=0)
    #
    #
    #         # sp_cam --> (sp_num, cls_num)
    #         # sp_features --> (sp_num, dim)
    #         # rgb_features --> (sp_num, 3)
    #
    #         # make similarity
    #         aff_mat = torch.nn.functional.cosine_similarity(sp_features.unsqueeze(0), sp_features.unsqueeze(1), dim=-1)
    #
    #         aff_mat = aff_mat * (torch.tensor(adjacent_matrix,dtype=torch.float32)+torch.eye(adjacent_matrix.shape[0])).to(device)
    #         trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    #
    #         RW_cams = torch.matmul(torch.tensor(sp_cams.t(),dtype=torch.float32), torch.tensor(trans_mat,dtype=torch.float32)) # cls_num, sp_num
    #         RW_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_cams.dtype).to(device)
    #         RW_class_response_maps = RW_class_response_maps.cpu()
    #         RW_cams = RW_cams.cpu()
    #         class_response_maps = class_response_maps.cpu()
    #
    #         for cls_idx in lables:
    #             for idx, temp_label in enumerate(range(sp_num)):
    #                 sp_mask = sp_label == temp_label
    #                 sp_mask_tensor = torch.tensor(sp_mask)
    #                 RW_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_cams[cls_idx,temp_label] # single batch input
    #
    #         RW_class_response_maps = F.upsample(RW_class_response_maps,size=(cam_H,cam_W),mode='bilinear', align_corners=True)
    #
    #         # a = 0.5
    #         # smooth_class_response_maps = a * RW_class_response_maps+(1-a)*class_response_maps
    #
    #         # cam_valid_peak_list, cam_peak_score = self.get_peak_list(class_response_maps,lables,peak_threshold)
    #         RW_valid_peak_list, RW_peak_score = self.get_peak_list(RW_class_response_maps,lables,peak_threshold)
    #         # smooth_valid_peak_list, smooth_peak_score = self.get_peak_list(smooth_class_response_maps, lables, peak_threshold)
    #
    #         # if len(smooth_peak_score) > 0:
    #         #     valid_peak_list = torch.stack(smooth_valid_peak_list)
    #         #     peak_score = torch.tensor(smooth_peak_score)
    #         #     # classification confidence scores, class-aware and instance-aware visual cues
    #         #     return smooth_class_response_maps, valid_peak_list, peak_score
    #         if len(RW_peak_score) > 0:
    #             valid_peak_list = torch.stack(RW_valid_peak_list)
    #             peak_score = torch.tensor(RW_peak_score)
    #             # classification confidence scores, class-aware and instance-aware visual cues
    #             return RW_class_response_maps, valid_peak_list, peak_score
    #         else:
    #             return None

    def train(self, mode=True):
        super(PeakResponseMapping_aff, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super(PeakResponseMapping_aff, self).train(False)
        self._patch()
        self.inferencing = True
        return self

    def label2onehot(self, cls_label, cls_num):
        onehot_label = np.zeros(cls_num)
        for cls_idx in cls_label:
            onehot_label[cls_idx] = 1

        return onehot_label

    def get_peak_list(self, class_response_maps, lables, peak_threshold, peak_filter=None):
        if self.enable_peak_stimulation:
            # sub-pixel peak finding
            if self.sub_pixel_locating_factor > 1:
                class_response_maps = F.upsample(class_response_maps, scale_factor=self.sub_pixel_locating_factor,
                                                 mode='bilinear', align_corners=True)
            # aggregate responses from informative receptive fields estimated via class peak responses
            if peak_filter == None:
                peak_list = peak_stimulation_aff(class_response_maps, win_size=self.win_size,
                                                 peak_filter=self.peak_filter)
            else:
                peak_list = peak_stimulation_aff(class_response_maps, win_size=self.win_size, peak_filter=peak_filter)
        else:
            # aggregate responses from all receptive fields
            peak_list = F.adaptive_avg_pool2d(class_response_maps, 1).squeeze(2).squeeze(2)

        # extract instance-aware visual cues, i.e., peak response maps
        assert class_response_maps.size(
            0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'
        if peak_list is None:
            if peak_filter == None:
                peak_list = peak_stimulation_aff(class_response_maps, return_aggregation=False, win_size=self.win_size,
                                                 peak_filter=self.peak_filter)
            else:
                peak_list = peak_stimulation_aff(class_response_maps, return_aggregation=False, win_size=self.win_size,
                                                 peak_filter=peak_filter)
            # peak_list = peak_stimulation_aff(class_response_maps, return_aggregation=False, win_size=self.win_size,peak_filter=self.peak_filter)

        valid_peak_list = []
        peak_score = []
        for cls_idx in lables:
            peak_list_cls = peak_list[peak_list[:, 1] == cls_idx]
            # print(peak_list_cls)
            exist = 0
            for idx in range(peak_list_cls.size(0)):
                peak_val = class_response_maps[
                    peak_list_cls[idx, 0], peak_list_cls[idx, 1], peak_list_cls[idx, 2], peak_list_cls[idx, 3]]
                # print(peak_val)
                if peak_val > peak_threshold:
                    exist = 1
                    # starting from the peak
                    valid_peak_list.append(peak_list_cls[idx, :])
                    peak_score.append(peak_val)
            # if exist == 0:
            #     peak_val_list = [class_response_maps[0, cls_idx, peak_list_cls[idx, 2], peak_list_cls[idx, 3]] for
            #                      idx in range(peak_list_cls.size(0))]
            #     peak_val_list = torch.tensor(peak_val_list)
            #     peak_val = class_response_maps[peak_list_cls[peak_val_list.argmax(), 0], \
            #                                    peak_list_cls[peak_val_list.argmax(), 1], \
            #                                    peak_list_cls[peak_val_list.argmax(), 2], \
            #                                    peak_list_cls[peak_val_list.argmax(), 3]]
            #     valid_peak_list.append(peak_list_cls[peak_val_list.argmax(), :])
            #     peak_score.append(peak_val)

        return valid_peak_list, peak_score


def peak_response_mapping_aff_vis(
        backbone: nn.Module = fc_resnet50_aff(),
        enable_peak_stimulation: bool = True,
        enable_peak_backprop: bool = True,
        win_size: int = 3,
        sub_pixel_locating_factor: int = 8,
        filter_type: Union[str, int, float] = 'median') -> nn.Module:
    """Peak Response Mapping.
    """

    model = PeakResponseMapping_aff_visual(
        backbone,
        enable_peak_stimulation=enable_peak_stimulation,
        enable_peak_backprop=enable_peak_backprop,
        win_size=win_size,
        sub_pixel_locating_factor=sub_pixel_locating_factor,
        filter_type=filter_type)
    return model


# only for inferencing visual
class PeakResponseMapping_aff_visual(nn.Sequential):

    def __init__(self, *args, **kargs):
        super(PeakResponseMapping_aff_visual, self).__init__(*args)

        self.inferencing = True
        # use global average pooling to aggregate responses if peak stimulation is disabled
        self.enable_peak_stimulation = kargs.get('enable_peak_stimulation', True)
        # return only the class response maps in inference mode if peak backpropagation is disabled
        self.enable_peak_backprop = kargs.get('enable_peak_backprop', True)
        # window size for peak finding
        self.win_size = kargs.get('win_size', 3)
        # sub-pixel peak finding
        self.sub_pixel_locating_factor = kargs.get('sub_pixel_locating_factor', 1)
        # peak filtering
        self.filter_type = kargs.get('filter_type', 'median')
        if self.filter_type == 'median':
            self.peak_filter = self._median_filter
        elif self.filter_type == 'mean':
            self.peak_filter = self._mean_filter
        elif self.filter_type == 'max':
            self.peak_filter = self._max_filter
        elif isinstance(self.filter_type, (int, float)):
            self.peak_filter = lambda x: self.filter_type
        else:
            self.peak_filter = None

    @staticmethod
    def _median_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _mean_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _max_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _lzc_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return 0.5 * threshold.contiguous().view(batch_size, num_channels, 1, 1)

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward

    def instance_nms(self, instance_list, threshold=0.3, merge_peak_response=True):
        selected_instances = []
        while len(instance_list) > 0:
            instance = instance_list.pop(0)
            selected_instances.append(instance)
            src_mask = instance[2].astype(bool)
            src_peak_response = instance[3]

            def iou_filter(x):
                dst_mask = x[2].astype(bool)
                # IoU
                intersection = np.logical_and(src_mask, dst_mask).sum()
                union = np.logical_or(src_mask, dst_mask).sum()
                iou = intersection / (union + 1e-10)
                if iou < threshold:
                    return x
                else:
                    if merge_peak_response:
                        nonlocal src_peak_response
                        src_peak_response += x[3]
                    return None

            instance_list = list(filter(iou_filter, instance_list))
        return selected_instances

    def instance_seg(self, class_response_maps, peak_list, peak_response_maps, retrieval_cfg):
        # cast tensors to numpy array
        class_response_maps = class_response_maps.squeeze().cpu().numpy()
        peak_list = peak_list.cpu().numpy()
        peak_response_maps = peak_response_maps.cpu().numpy()

        img_height, img_width = peak_response_maps.shape[1], peak_response_maps.shape[2]

        # image size
        img_area = img_height * img_width

        # segment proposals off-the-shelf
        proposals = retrieval_cfg['proposals']

        # proposal contour width
        contour_width = retrieval_cfg.get('contour_width', 5)

        # limit range of proposal size
        proposal_size_limit = retrieval_cfg.get('proposal_size_limit', (0.00002, 0.85))

        # selected number of proposals
        proposal_count = retrieval_cfg.get('proposal_count', 100)

        # nms threshold
        nms_threshold = retrieval_cfg.get('nms_threshold', 0.3)

        # merge peak response during nms
        merge_peak_response = retrieval_cfg.get('merge_peak_response', True)

        # metric free parameters
        param = retrieval_cfg.get('param', None)

        # process each peak
        instance_list = []
        for i in range(len(peak_response_maps)):
            class_idx = peak_list[i, 1]

            # extract hyper-params
            if isinstance(param, tuple):
                # shared param
                bg_threshold_factor, penalty_factor, balance_factor = param
            elif isinstance(param, list):
                # independent params between classes
                bg_threshold_factor, penalty_factor, balance_factor = param[class_idx]
            else:
                raise TypeError('Invalid hyper-params "%s".' % param)

            class_response = imresize(class_response_maps[class_idx], (img_height, img_width), interp='bicubic')
            bg_response = (class_response < bg_threshold_factor * class_response.mean()).astype(np.float32)
            peak_response_map = peak_response_maps[i]

            # select proposal
            max_val = -np.inf
            instance_mask = None

            for j in range(min(proposal_count, len(proposals))):
                raw_mask = imresize(proposals[j].astype(int), peak_response_map.shape, interp='nearest')
                # get contour of the proposal
                contour_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_GRADIENT,
                                                np.ones((contour_width, contour_width), np.uint8)).astype(bool)
                mask = raw_mask.astype(bool)
                # metric
                mask_area = mask.sum()
                if (mask_area >= proposal_size_limit[1] * img_area) or \
                        (mask_area < proposal_size_limit[0] * img_area):
                    continue
                else:
                    val = balance_factor * peak_response_map[mask].sum() + \
                          peak_response_map[contour_mask].sum() - \
                          penalty_factor * bg_response[mask].sum()
                    if val > max_val:
                        max_val = val
                        instance_mask = mask

            if instance_mask is not None:
                instance_list.append((max_val, class_idx, instance_mask, peak_response_map))

        instance_list = sorted(instance_list, key=lambda x: x[0], reverse=True)
        if nms_threshold is not None:
            instance_list = self.instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), nms_threshold,
                                              merge_peak_response)
        return [dict(score=v[0], category=v[1], mask=v[2], prm=v[3]) for v in instance_list]

    # use proposal score to replace cam score
    # draw
    def forward(self, input, lables, img, class_threshold=0, peak_threshold=10, retrieval_cfg=None, graph_op=None,
                resolution=7):
        assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
        dir = "./voc_visual"
        if not os.path.exists(dir):
            os.makedirs(dir)
        with torch.no_grad():

            # _, _, H, W = input.shape  # batchsize == 1
            H, W, _ = np.array(img).shape

            LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            HSV_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # classification network forwarding
            class_response_maps, p1, p2, p3, p4 = super(PeakResponseMapping_aff_visual, self).forward(input)
            class_response_maps = class_response_maps.detach()
            org_size_class_response_maps = F.upsample(class_response_maps, size=(H, W), mode='bilinear',
                                                      align_corners=True)

            _, cls_num, cam_H, cam_W = class_response_maps.shape

            onehot_label = self.label2onehot(lables, cls_num)
            class_response_maps = class_response_maps * torch.tensor(onehot_label).view(1, cls_num, 1, 1)

            _, _, f_H_p1, f_W_p1 = p1.shape
            _, _, f_H_p2, f_W_p2 = p2.shape
            _, _, f_H_p3, f_W_p3 = p3.shape
            _, _, f_H_p4, f_W_p4 = p4.shape

            Align_op_p1 = RoIAlign((resolution, resolution), f_H_p1 / H, 0)
            Align_op_p2 = RoIAlign((resolution, resolution), f_H_p2 / H, 0)
            Align_op_p3 = RoIAlign((resolution, resolution), f_H_p3 / H, 0)
            Align_op_p4 = RoIAlign((resolution, resolution), f_H_p4 / H, 0)

            # graph operate
            sp_label = graph_op["sp_label"]
            adjacent_matrix = graph_op["adjacent_matrix"]
            img_id = graph_op["img_id"]
            belong_mat = graph_op["belong_mat"]
            proposal_score = graph_op["proposal_score"]
            cocoGT = graph_op["cocoGT"]
            proposal_sum = graph_op["proposal_sum"]
            proposal = graph_op["proposal"]
            print(img_id)

            # add bg_score
            # bg_score = 1 - proposal_score.sum(-1)
            # proposal_score = np.concatenate((bg_score[:,None],proposal_score),axis=1)
            #
            # sp_scores = np.zeros((adjacent_matrix.shape[0],proposal_score.shape[-1])) # sp_num, cls_num + 1
            # for row_idx in range(belong_mat.shape[0]):
            #     score = proposal_score[row_idx]
            #     belong_row = belong_mat[row_idx] != 0
            #
            #     sp_scores[belong_row] += score
            #
            # sp_scores = torch.tensor(sp_scores)
            # sp_scores_normalize = torch.nn.functional.normalize(sp_scores,p=1,dim=-1)

            # add bg_score
            # bg_score = 1 - proposal_score.sum(-1)
            # proposal_score = np.concatenate((bg_score[:, None], proposal_score), axis=1)

            # max_idx = np.argmax(proposal_score, axis=-1)
            # flag = np.zeros_like(max_idx)
            # for label in lables:
            #     flag += (label + 1).item() == max_idx
            #
            # bg_score_const = np.zeros(proposal_score.shape[-1])
            # bg_score_const[0] = 1
            # proposal_score[flag == 0] = bg_score_const
            #
            # sp_scores = np.zeros((adjacent_matrix.shape[0], proposal_score.shape[-1]))  # sp_num, cls_num + 1
            # for row_idx in range(belong_mat.shape[0]):
            #     score = proposal_score[row_idx]
            #     belong_row = belong_mat[row_idx] != 0
            #
            #     sp_scores[belong_row] += score
            #
            # sp_scores = torch.tensor(sp_scores)
            # sp_scores_normalize = torch.nn.functional.normalize(sp_scores, p=1, dim=-1)

            low_label = np.min(sp_label)
            high_label = np.max(sp_label)

            # for idx, temp_label in tqdm(enumerate(range(low_label, high_label + 1)),total=adjacent_matrix.shape[0]):
            #     sp_mask = sp_label == temp_label
            #     sp_mask_tensor = torch.tensor(sp_mask)
            #
            #     if idx == 0:
            #         sp_proposal = sp_mask[None,:,:]
            #     else:
            #         sp_proposal = np.concatenate((sp_proposal, sp_mask[None,:,:]),axis=0)
            #     cam_scores = (sp_mask_tensor * org_size_class_response_maps).sum((-2,-1)) / sp_mask_tensor.sum() # --> (1, cls_num)
            #
            #
            #     ind_xy = np.nonzero(sp_mask)
            #     xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
            #
            #     mask = sp_mask[ymin:ymax, xmin:xmax]
            #     mask = imresize(mask.astype(int), (resolution, resolution), interp='nearest')  # (7,7)
            #     mask = torch.tensor(mask)
            #
            #     box = torch.tensor([[0, xmin, ymin, xmax, ymax]],dtype=torch.float32)
            #     box_feature_p1 = Align_op_p1(p1, box)  # (1,7,7)
            #     box_feature_p2 = Align_op_p2(p2, box)  # (1,7,7)
            #     box_feature_p3 = Align_op_p3(p3, box)  # (1,7,7)
            #     box_feature_p4 = Align_op_p4(p4, box)  # (1,7,7)
            #
            #     sp_feature_p1 = get_sp_f(box_feature_p1,mask,resolution)
            #     sp_feature_p2 = get_sp_f(box_feature_p2,mask,resolution)
            #     sp_feature_p3 = get_sp_f(box_feature_p3,mask,resolution)
            #     sp_feature_p4 = get_sp_f(box_feature_p4,mask,resolution)
            #
            #     if idx == 0:
            #         sp_cams = cam_scores
            #         sp_features_p1 = sp_feature_p1
            #         sp_features_p2 = sp_feature_p2
            #         sp_features_p3 = sp_feature_p3
            #         sp_features_p4 = sp_feature_p4
            #
            #         RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
            #         LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
            #         HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
            #         for color_idx in range(1,3):
            #             temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
            #             temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
            #             temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
            #
            #             RGB_feature = np.concatenate((RGB_feature,temp_feature_rgb),axis=0)
            #             LAB_feature = np.concatenate((LAB_feature,temp_feature_lab),axis=0)
            #             HSV_feature = np.concatenate((HSV_feature,temp_feature_hsv),axis=0)
            #
            #         RGB_features = RGB_feature.transpose()
            #         LAB_features = LAB_feature.transpose()
            #         HSV_features = HSV_feature.transpose()
            #
            #     else:
            #         sp_cams = torch.cat((sp_cams, cam_scores), dim=0)
            #         sp_features_p1 = torch.cat((sp_features_p1, sp_feature_p1),dim=0)
            #         sp_features_p2 = torch.cat((sp_features_p2, sp_feature_p2),dim=0)
            #         sp_features_p3 = torch.cat((sp_features_p3, sp_feature_p3),dim=0)
            #         sp_features_p4 = torch.cat((sp_features_p4, sp_feature_p4),dim=0)
            #
            #         RGB_feature = cv2.calcHist([img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
            #         LAB_feature = cv2.calcHist([LAB_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
            #         HSV_feature = cv2.calcHist([HSV_img], [0], np.uint8(sp_mask * 255), [16], [0, 255])
            #         for color_idx in range(1, 3):
            #             temp_feature_rgb = cv2.calcHist([img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
            #             temp_feature_lab = cv2.calcHist([LAB_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
            #             temp_feature_hsv = cv2.calcHist([HSV_img], [color_idx], np.uint8(sp_mask * 255), [16], [0, 255])
            #
            #             RGB_feature = np.concatenate((RGB_feature, temp_feature_rgb), axis=0)
            #             LAB_feature = np.concatenate((LAB_feature, temp_feature_lab), axis=0)
            #             HSV_feature = np.concatenate((HSV_feature, temp_feature_hsv), axis=0)
            #
            #         RGB_features = np.concatenate((RGB_features,RGB_feature.transpose()),axis=0)
            #         LAB_features = np.concatenate((LAB_features,LAB_feature.transpose()),axis=0)
            #         HSV_features = np.concatenate((HSV_features,HSV_feature.transpose()),axis=0)

            # RGB_features = torch.tensor(RGB_features)
            # LAB_features = torch.tensor(LAB_features)
            # HSV_features = torch.tensor(HSV_features)
            #
            # # make similarity
            # aff_sp_f = torch.nn.functional.cosine_similarity(sp_features_p1.unsqueeze(0), sp_features_p1.unsqueeze(1), dim=-1) * \
            #            torch.nn.functional.cosine_similarity(sp_features_p2.unsqueeze(0), sp_features_p2.unsqueeze(1),dim=-1) * \
            #            torch.nn.functional.cosine_similarity(sp_features_p3.unsqueeze(0), sp_features_p3.unsqueeze(1),dim=-1) * \
            #            torch.nn.functional.cosine_similarity(sp_features_p4.unsqueeze(0), sp_features_p4.unsqueeze(1),dim=-1)
            # # aff_sp_f = torch.nn.functional.cosine_similarity(sp_features_p4.unsqueeze(0), sp_features_p4.unsqueeze(1),dim=-1)
            #
            # aff_score = torch.nn.functional.cosine_similarity(sp_scores.unsqueeze(0), sp_scores.unsqueeze(1), dim=-1)
            #
            # aff_color = torch.nn.functional.cosine_similarity(RGB_features.unsqueeze(0), RGB_features.unsqueeze(1), dim=-1) * \
            #             torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1), dim=-1) * \
            #             torch.nn.functional.cosine_similarity(HSV_features.unsqueeze(0), HSV_features.unsqueeze(1), dim=-1)
            # # aff_color = torch.nn.functional.cosine_similarity(LAB_features.unsqueeze(0), LAB_features.unsqueeze(1), dim=-1)
            #
            # aff_mat = aff_sp_f + aff_color + aff_score
            # # aff_mat = aff_sp_f * aff_color
            # # aff_mat = aff_sp_f * aff_score * aff_rgb
            #
            # aff_mat_mask = aff_mat * (torch.tensor(adjacent_matrix, dtype=torch.float32) + torch.eye(adjacent_matrix.shape[0]))
            #
            # trans_mat = aff_mat_mask / (torch.sum(aff_mat_mask, dim=0, keepdim=True)+1e-5)
            # # D = np.diag(aff_mat_mask.sum(-1))
            # # D_1 = torch.tensor(np.sqrt(np.linalg.pinv(D)))
            # # trans_mat = D_1 @ aff_mat_mask @ D_1
            #
            # plt.close()
            # fig,axx = plt.subplots(2, 3)
            #
            # sns.heatmap(aff_sp_f.numpy(),cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,0])
            # axx[0,0].set_title("aff_sp_f")
            #
            # sns.heatmap(aff_score.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,1])
            # axx[0,1].set_title("aff_score")
            #
            # # axx[0,2].axis("off")
            # sns.heatmap(aff_color.numpy(),cmap="jet", xticklabels=False, yticklabels=False, cbar=False,ax = axx[0,2])
            # axx[0,2].set_title("aff_color")
            #
            # sns.heatmap(aff_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,0])
            # axx[1,0].set_title("aff_mat")
            #
            # sns.heatmap(aff_mat_mask.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,1])
            # axx[1,1].set_title("aff_mat_mask")
            #
            # sns.heatmap(trans_mat.numpy(), cmap="jet",xticklabels=False, yticklabels=False, cbar=False,ax = axx[1,2])
            # axx[1,2].set_title("trans_mat")
            #
            # print(os.path.join(dir,str(img_id)+"_aff.jpg"))
            #
            # plt.savefig(os.path.join(dir,str(img_id)+"_aff.jpg"), dpi=300)

            # # 包括背景类 --> (21, sp_num)
            # # RW_cams = torch.tensor(sp_cams.t(), dtype=torch.float32)
            # RW_cams = torch.tensor(sp_scores_normalize.t(), dtype=torch.float32)
            # trans_mat = torch.tensor(trans_mat, dtype=torch.float32)

            cam_valid_peak_list, cam_peak_score = self.get_peak_list(class_response_maps, lables, peak_threshold)
            response_maps = [class_response_maps]
            valid_peak_list = [cam_valid_peak_list]
            scale = 0.7

            # RW_cams = torch.matmul(torch.tensor(trans_mat.t(),dtype=torch.float32),torch.tensor(sp_cams,dtype=torch.float32))
            # RW_cams = RW_cams.t()

            iter_TIME = 4 + 1

            # for i in range(1):
            #     if i == 0:
            #         RW_cams = RW_cams
            #     else:
            #         RW_cams =  torch.matmul(RW_cams, trans_mat)
            #     RW_cams = torch.nn.functional.normalize(RW_cams, p=1, dim=0)
            #
            #     RW_class_response_maps = torch.zeros_like(org_size_class_response_maps,dtype=RW_cams.dtype)
            #
            #     for cls_idx in lables:
            #         for idx, temp_label in enumerate(range(low_label, high_label + 1)):
            #             sp_mask = sp_label == temp_label
            #             sp_mask_tensor = torch.tensor(sp_mask)
            #             # exclusive bg class
            #             RW_class_response_maps[0, cls_idx,sp_mask_tensor] = RW_cams[cls_idx+1,idx] # single batch input
            #
            #     RW_class_response_maps = F.upsample(RW_class_response_maps,size=(cam_H,cam_W),mode='bilinear', align_corners=True)
            #
            #     RW_valid_peak_list, RW_peak_score = self.get_peak_list(RW_class_response_maps,lables,0,self._lzc_filter)
            #
            #     response_maps.append(RW_class_response_maps)
            #     valid_peak_list.append(RW_valid_peak_list)
            #
            # RW_class_response_maps = RW_class_response_maps.numpy()
            # RW_class_response_maps = RW_class_response_maps - np.min(RW_class_response_maps,(-2,-1),keepdims = True)
            # RW_class_response_maps = RW_class_response_maps / np.max(RW_class_response_maps,(-2,-1),keepdims = True)
            # RW_class_response_maps = torch.tensor(RW_class_response_maps)

            # # print(RW_class_response_maps)
            # mask_CAM = RW_class_response_maps * class_response_maps
            # mask_CAM_valid_peak_list, mask_CAM_peak_score = self.get_peak_list(mask_CAM, lables, peak_threshold)
            # response_maps.append(mask_CAM)
            # valid_peak_list.append(mask_CAM_valid_peak_list)

            cam_num = onehot_label.sum().item()
            plt.close()
            # f, axarr = plt.subplots(len(response_maps), int(cam_num + 1))
            # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.1, hspace=0.1)

            # print(proposal.shape)
            # for proposal_idx,propo in enumerate(proposal):
            #     # heatmap = cv2.applyColorMap(np.uint8(255. * propo), cv2.COLORMAP_JET)
            #     # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            #     # result = heatmap * scale + (img) * (1 - scale)
            #     # plt.close()
            #     # plt.imshow(np.uint8(result))
            #     # plt.axis("off")
            #     # if not os.path.exists(os.path.join(dir,str(img_id))):
            #     #     os.makedirs(os.path.join(dir,str(img_id)))
            #     # plt.savefig(os.path.join(dir,str(img_id),"proposal{}".format(proposal_idx) + ".jpg"), dpi=300, bbox_inches='tight',pad_inches=0)
            #
            #     plt.close()
            #     plt.imshow(np.uint8(255 * np.concatenate((propo[...,None],propo[...,None],propo[...,None]),axis=-1)))
            #     plt.axis("off")
            #     if not os.path.exists(os.path.join(dir, str(img_id))):
            #         os.makedirs(os.path.join(dir, str(img_id)))
            #     plt.savefig(os.path.join(dir, str(img_id), "proposal{}".format(proposal_idx) + ".jpg"), dpi=300,
            #                 bbox_inches='tight', pad_inches=0)

            HEATMAP = []
            for idx in range(1):
                if idx == 0:
                    img_col = cv2.resize(img, (
                    int(cam_W * self.sub_pixel_locating_factor), int(cam_H * self.sub_pixel_locating_factor)))

                    plt.close()
                    plt.imshow(np.uint8(img))
                    plt.axis("off")

                    plt.savefig(os.path.join(dir, str(img_id) + "org" + ".jpg"), dpi=300, bbox_inches='tight',
                                pad_inches=0)

                elif idx == 1:
                    sp_label = cv2.resize(np.uint8(sp_label), (
                    int(cam_W * self.sub_pixel_locating_factor), int(cam_H * self.sub_pixel_locating_factor)),
                                          cv2.INTER_NEAREST)
                    res = mark_boundaries(img_col, sp_label) * 255
                    img_col = np.concatenate((img_col, res), axis=1)

                elif idx == 2:
                    polygons = []
                    box_list = []
                    label_list = []
                    gt_ann_ids = cocoGT.getAnnIds(imgIds=[img_id])
                    gt_anns = cocoGT.loadAnns(gt_ann_ids)
                    print(gt_anns)
                    for ann in gt_anns:
                        if 'segmentation' in ann:
                            # print("here")
                            # print(ann['segmentation'])
                            rle = maskUtils.frPyObjects(ann['segmentation'], img.shape[0], img.shape[1])
                            mask = maskUtils.decode(rle).transpose(2, 0, 1)
                            # print(mask)
                            # print(mask.shape)

                            polygons.append(mask)
                        label_list.append(ann['category_id'] - 1)
                        box_list.append(np.array(ann['bbox']))

                    polygons = np.concatenate(polygons, axis=0)
                    # print(polygons.shape)
                    label_list = np.array(label_list)
                    box_list = np.concatenate(box_list, axis=0).reshape((-1, 4))
                    # gt_img = imshow_det_bboxes(img,box_list,labels=label_list,segms=polygons,show=False)
                    gt_img = imshow_det_bboxes(img, labels=label_list, segms=polygons, show=False)

                    img_col = np.concatenate((img_col, cv2.resize(gt_img, (
                    int(cam_W * self.sub_pixel_locating_factor), int(cam_H * self.sub_pixel_locating_factor)))), axis=1)

                elif idx == 3:
                    print(proposal_sum.shape)
                    print(proposal_sum)
                    proposal_sum = np.uint8(255 * proposal_sum / proposal_sum.max())
                    heatmap = cv2.applyColorMap(proposal_sum, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    res = heatmap * scale + (img) * (1 - scale)

                    plt.close()
                    plt.imshow(np.uint8(res))
                    plt.axis("off")
                    plt.savefig(os.path.join(dir, str(img_id) + "proposal_sum" + ".jpg"), dpi=300, bbox_inches='tight',
                                pad_inches=0)

                    res = cv2.resize(res, (
                    int(cam_W * self.sub_pixel_locating_factor), int(cam_H * self.sub_pixel_locating_factor)))
                    img_col = np.concatenate((img_col, res), axis=1)

                else:
                    img_col = np.concatenate((img_col, 255 * np.ones(
                        (int(cam_W * self.sub_pixel_locating_factor), int(cam_H * self.sub_pixel_locating_factor), 3))),
                                             axis=1)

                for i, cls_idx in enumerate(lables):
                    CRM = response_maps[idx]

                    cam = CRM[0, cls_idx].detach().cpu().numpy()
                    cam = cam - np.min(cam)
                    cam = cam / np.max(cam)
                    cam = np.uint8(255 * cam)

                    heatmap = cv2.applyColorMap(cv2.resize(cam, (W, H)), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    result = heatmap * scale + (img) * (1 - scale)
                    result = np.uint8(result)
                    result = cv2.resize(result, (
                    int(cam_W * self.sub_pixel_locating_factor), int(cam_H * self.sub_pixel_locating_factor)))

                    VPL = valid_peak_list[idx]

                    for peak in VPL:
                        if peak[1] == cls_idx:
                            result = cv2.circle(result, center=(int(peak[3]), int(peak[2])), radius=2,
                                                color=(0, 0, 255), thickness=-1)

                    if idx == 0:
                        heat = result

                        plt.close()
                        plt.imshow(np.uint8(heat))
                        plt.axis("off")

                        plt.savefig(os.path.join(dir, str(img_id), str(img_id) + "heat" + ".jpg"), dpi=300,
                                    bbox_inches='tight', pad_inches=0)

                        print(len(VPL))
                        for idk, peak in enumerate(VPL):
                            if peak[1] == cls_idx:
                                result = cv2.circle(result, center=(int(peak[3]), int(peak[2])), radius=2,
                                                    color=(0, 0, 255), thickness=-1)
                                # mask_proposals: 200,375,500
                                x = int(peak[2] * proposal.shape[1] / 112)
                                y = int(peak[3] * proposal.shape[2] / 112)

                                avgmask = proposal[proposal[:, x, y] > 0, :, :].mean(0) > 0.7

                                heatmap = cv2.applyColorMap(np.uint8(avgmask * 1. * 255), cv2.COLORMAP_JET)
                                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                                res = heatmap * scale + (img) * (1 - scale)

                                plt.close()
                                plt.imshow(np.uint8(res))
                                plt.axis("off")
                                plt.savefig(os.path.join(dir, str(img_id), "avg_mask{}".format(idk) + ".jpg"), dpi=300,
                                            bbox_inches='tight', pad_inches=0)

                    if i == 0:
                        heat = result
                        plt.close()
                        result = cv2.resize(result, (
                        int(cam_W * self.sub_pixel_locating_factor), int(cam_H * self.sub_pixel_locating_factor)))

                        plt.imshow(np.uint8(result))
                        plt.axis("off")
                        plt.savefig(os.path.join(dir, str(img_id), "heat_{}".format(idk) + ".jpg"), dpi=300,
                                    bbox_inches='tight', pad_inches=0)

                    else:
                        heat = np.concatenate((heat, result), axis=0)

                HEATMAP.append(heat)

            HEATMAP = np.concatenate(HEATMAP, axis=1)
            result = np.concatenate((img_col, HEATMAP), axis=0)
            result = np.concatenate((255 * np.ones((result.shape[0], 80, 3)), result), axis=1)

            for i, cls_idx in enumerate(lables):
                cv2.putText(result, VOC_CLASSES[cls_idx], (5, (i) * 120 + 180), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (255, 0, 0), 1)

            plt.imshow(np.uint8(result))
            plt.axis("off")
            plt.savefig(os.path.join(dir, str(img_id), str(img_id) + ".jpg"), dpi=300, bbox_inches='tight')

            from pycocotools import mask as COCOMask
            import json

            def coco_encode(mask):
                encoding = COCOMask.encode(np.asfortranarray(mask))
                encoding['counts'] = encoding['counts'].decode('utf-8')
                return encoding

            # proposal_list = [31,119,146,13,104,]
            # proposal_list = [23,5,156,18]
            # proposal_list = [23, 97] # person
            proposal_list = [75, 169]

            json_results = []
            for proposal_idx in proposal_list:
                data = dict()
                data['image_id'] = int(img_id)
                data['segmentation'] = coco_encode(proposal[proposal_idx].astype(np.uint8))
                data['score'] = float(1)
                data['category_id'] = 1
                json_results.append(data)

            with open(os.path.join(dir, str(img_id), str(img_id) + ".json"), 'w') as f:
                f.write(json.dumps(json_results))

            # if len(peak_score) > 0:
            #     valid_peak_list = torch.stack(valid_peak_list)
            #     peak_score = torch.tensor(peak_score)
            #     # classification confidence scores, class-aware and instance-aware visual cues
            #     return class_response_maps, valid_peak_list, peak_score
            # else:
            #     return None

    def train(self, mode=True):
        super(PeakResponseMapping_aff_visual, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super(PeakResponseMapping_aff_visual, self).train(False)
        self._patch()
        self.inferencing = True
        return self

    def label2onehot(self, cls_label, cls_num):
        onehot_label = np.zeros(cls_num)
        for cls_idx in cls_label:
            onehot_label[cls_idx] = 1

        return onehot_label

    def get_peak_list(self, class_response_maps, lables, peak_threshold, peak_filter=None):
        if self.enable_peak_stimulation:
            # sub-pixel peak finding
            if self.sub_pixel_locating_factor > 1:
                class_response_maps = F.upsample(class_response_maps, scale_factor=self.sub_pixel_locating_factor,
                                                 mode='bilinear', align_corners=True)
            # aggregate responses from informative receptive fields estimated via class peak responses
            if peak_filter == None:
                peak_list = peak_stimulation_aff(class_response_maps, win_size=self.win_size,
                                                 peak_filter=self.peak_filter)
            else:
                peak_list = peak_stimulation_aff(class_response_maps, win_size=self.win_size, peak_filter=peak_filter)
        else:
            # aggregate responses from all receptive fields
            peak_list = F.adaptive_avg_pool2d(class_response_maps, 1).squeeze(2).squeeze(2)

        # extract instance-aware visual cues, i.e., peak response maps
        assert class_response_maps.size(
            0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'
        if peak_list is None:
            if peak_filter == None:
                peak_list = peak_stimulation_aff(class_response_maps, return_aggregation=False, win_size=self.win_size,
                                                 peak_filter=self.peak_filter)
            else:
                peak_list = peak_stimulation_aff(class_response_maps, return_aggregation=False, win_size=self.win_size,
                                                 peak_filter=peak_filter)
            # peak_list = peak_stimulation_aff(class_response_maps, return_aggregation=False, win_size=self.win_size,peak_filter=self.peak_filter)

        valid_peak_list = []
        peak_score = []
        for cls_idx in lables:
            peak_list_cls = peak_list[peak_list[:, 1] == cls_idx]
            # print(peak_list_cls)
            exist = 0
            for idx in range(peak_list_cls.size(0)):
                peak_val = class_response_maps[
                    peak_list_cls[idx, 0], peak_list_cls[idx, 1], peak_list_cls[idx, 2], peak_list_cls[idx, 3]]
                # print(peak_val)
                if peak_val > peak_threshold:
                    exist = 1
                    # starting from the peak
                    valid_peak_list.append(peak_list_cls[idx, :])
                    peak_score.append(peak_val)
            # if exist == 0:
            #     peak_val_list = [class_response_maps[0, cls_idx, peak_list_cls[idx, 2], peak_list_cls[idx, 3]] for
            #                      idx in range(peak_list_cls.size(0))]
            #     peak_val_list = torch.tensor(peak_val_list)
            #     peak_val = class_response_maps[peak_list_cls[peak_val_list.argmax(), 0], \
            #                                    peak_list_cls[peak_val_list.argmax(), 1], \
            #                                    peak_list_cls[peak_val_list.argmax(), 2], \
            #                                    peak_list_cls[peak_val_list.argmax(), 3]]
            #     valid_peak_list.append(peak_list_cls[peak_val_list.argmax(), :])
            #     peak_score.append(peak_val)

        return valid_peak_list, peak_score

