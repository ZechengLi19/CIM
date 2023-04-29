# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from typing import Union, Optional, List, Tuple


#%%
from types import MethodType
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from scipy.misc import imresize
from lib.prm.prm_modules import pr_conv2d, peak_stimulation

def imresize(arr,size,interp='bilibear',mode=None):
    im = Image.fromarray(np.uint8(arr),mode=mode)
    ts = type(size)
    if np.issubdtype(ts,np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size)*percent).astype(int))
    elif np.issubdtype(type(size),np.floating):
        size = tuple((np.array(im.size)*size).astype(int))
    else:
        size = (size[1],size[0])
    func = {'nearest':0,'lanczos':1,'biliear':2,'bicubic':3,'cubic':3}
    imnew = im.resize(size,resample=func[interp])    # 调用PIL库中的resize函数
    return np.array(imnew)

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
                contour_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_GRADIENT, np.ones((contour_width, contour_width), np.uint8)).astype(bool)
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
            instance_list = self.instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), nms_threshold, merge_peak_response)
        return [dict(score=v[0], category=v[1], mask=v[2], prm=v[3]) for v in instance_list]

    
    def forward(self, input, class_threshold=0, peak_threshold=10, retrieval_cfg=None):
        assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
        if self.inferencing:
            input.requires_grad_()

        # classification network forwarding
        
        class_response_maps = super(PeakResponseMapping, self).forward(input)

        if self.enable_peak_stimulation:
            # sub-pixel peak finding
            if self.sub_pixel_locating_factor > 1:
                class_response_maps = F.upsample(class_response_maps, scale_factor=self.sub_pixel_locating_factor, mode='bilinear', align_corners=True)
            # aggregate responses from informative receptive fields estimated via class peak responses
            peak_list, aggregation = peak_stimulation(class_response_maps, win_size=self.win_size, peak_filter=self.peak_filter)
        else:
            # aggregate responses from all receptive fields
            peak_list, aggregation = None, F.adaptive_avg_pool2d(class_response_maps, 1).squeeze(2).squeeze(2)

        if self.inferencing:
            if not self.enable_peak_backprop:
                # extract only class-aware visual cues
                return aggregation, class_response_maps
            
            # extract instance-aware visual cues, i.e., peak response maps
            assert class_response_maps.size(0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'
            if peak_list is None:
                peak_list = peak_stimulation(class_response_maps, return_aggregation=False, win_size=self.win_size, peak_filter=self.peak_filter)

            peak_response_maps = []
            valid_peak_list = []
            peak_score=[]
            # peak backpropagation
            grad_output = class_response_maps.new_empty(class_response_maps.size())
            for idx in range(peak_list.size(0)):
                if aggregation[peak_list[idx, 0], peak_list[idx, 1]] >= class_threshold:
                    peak_val = class_response_maps[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]]
                    if peak_val > peak_threshold:
                        grad_output.zero_()
                        # starting from the peak
                        grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]] = 1
                        if input.grad is not None:
                            input.grad.zero_()
                        class_response_maps.backward(grad_output, retain_graph=True)
                        prm = input.grad.detach().sum(1).clone().clamp(min=0)
                        peak_response_maps.append(prm / prm.sum())
                        valid_peak_list.append(peak_list[idx, :])
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
                return None # aggregation, class_response_maps # None   
        else:
            # classification confidence scores
            return aggregation

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
    

class FT_VGG(nn.Module):
    def __init__(self, model, num_classes):
        super(FT_VGG, self).__init__()
        # feature encoding
        self.features = model.features
        self.classifier = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def fc_vgg16( num_classes: int = 20, pretrained: bool = True):
    """FC ft_vgg16.
    """
    model = FT_VGG(models.vgg16(pretrained), num_classes)
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
        enable_peak_stimulation = enable_peak_stimulation,
        enable_peak_backprop = enable_peak_backprop, 
        win_size = win_size, 
        sub_pixel_locating_factor = sub_pixel_locating_factor, 
        filter_type = filter_type)
    return model




