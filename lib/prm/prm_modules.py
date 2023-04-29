# -*- coding: utf-8 -*-

# PeakStimulation
import torch
import torch.nn.functional as F
from torch.autograd import Function


class PeakStimulation(Function):

    @staticmethod
    def forward(ctx, input, return_aggregation, win_size, peak_filter):
        ctx.num_flags = 4

        # peak finding
        assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
        offset = (win_size - 1) // 2
        padding = torch.nn.ConstantPad2d(offset, float('-inf'))
        padded_maps = padding(input)
        batch_size, num_channels, h, w = padded_maps.size()
        element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
        element_map = element_map.to(input.device)
        _, indices  = F.max_pool2d(
            padded_maps,
            kernel_size = win_size, 
            stride = 1, 
            return_indices = True)
        peak_map = (indices == element_map)

        # peak filtering
        if peak_filter:
            mask = input >= peak_filter(input)
            peak_map = (peak_map & mask)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)
        
        # peak aggregation
        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
                peak_map.view(batch_size, num_channels, -1).sum(2)
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1)
        return (grad_input,) + (None,) * ctx.num_flags


def peak_stimulation(input, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation.apply(input, return_aggregation, win_size, peak_filter)

class PeakStimulation_aff(Function):
    @staticmethod
    def forward(ctx, input, return_aggregation, win_size, peak_filter):
        ctx.num_flags = 4

        # peak finding
        assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
        offset = (win_size - 1) // 2
        padding = torch.nn.ConstantPad2d(offset, float('-inf'))
        padded_maps = padding(input)
        batch_size, num_channels, h, w = padded_maps.size()
        element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
        element_map = element_map.to(input.device)
        _, indices  = F.max_pool2d(
            padded_maps,
            kernel_size = win_size,
            stride = 1,
            return_indices = True)
        peak_map = (indices == element_map)

        # peak filtering
        if peak_filter:
            mask = input >= peak_filter(input)
            peak_map = (peak_map & mask)
        peak_list = torch.nonzero(peak_map)

        return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1)
        return (grad_input,) + (None,) * ctx.num_flags


def peak_stimulation_aff(input, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation_aff.apply(input, return_aggregation, win_size, peak_filter)


# peak back
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.autograd import Function


class PreHook(Function):
    
    @staticmethod
    def forward(ctx, input, offset):
        ctx.save_for_backward(input, offset)
        return input.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, offset = ctx.saved_variables
        return (input - offset) * grad_output, None
    
class PostHook(Function):
    
    @staticmethod
    def forward(ctx, input, norm_factor):
        ctx.save_for_backward(norm_factor)
        return input.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        norm_factor, = ctx.saved_variables
        eps = 1e-10
        zero_mask = norm_factor < eps
        grad_input = grad_output / (torch.abs(norm_factor) + eps)
        grad_input[zero_mask.detach()] = 0
        return None, grad_input


def pr_conv2d(self, input):
    offset = input.min().detach()
    input = PreHook.apply(input, offset)
    resp = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups).detach()
    pos_weight = F.relu(self.weight).detach()
    norm_factor = F.conv2d(input - offset, pos_weight, None, self.stride, self.padding, self.dilation, self.groups)
    output = PostHook.apply(resp, norm_factor)
    return output
