# -*- coding: utf-8 -*-

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
    '''
    conv2d under excitation bp setting.
    '''
    offset = input.min().detach()
    input = PreHook.apply(input, offset)
    resp = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups).detach()
    pos_weight = F.relu(self.weight).detach()
    norm_factor = F.conv2d(input - offset, pos_weight, None, self.stride, self.padding, self.dilation, self.groups)
    output = PostHook.apply(resp, norm_factor)
    return output


class EBLinear(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output
    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        ### start EB-SPECIFIC CODE  ###
        # print("this is a {} linear layer ({})"
        #       .format('pos' if torch.use_pos_weights else 'neg', grad_output.sum().data[0]))
        weight = weight.clamp(min=0) if torch.use_pos_weights else weight.clamp(max=0).abs()

        input.data = input.data - input.data.min() if input.data.min() < 0 else input.data
        grad_output /= input.mm(weight.t()).abs() + 1e-10 # normalize
        ### stop EB-SPECIFIC CODE  ###

        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
            ### start EB-SPECIFIC CODE  ###
            grad_input *= input
            ### stop EB-SPECIFIC CODE  ###

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)


        return grad_input, grad_weight, grad_bias

def pr_linear(self, input):
    '''
    linear under excitation bp setting.
    '''
    output = EBLinear.apply(input, self.weight, self.bias)
    return output


from types import MethodType
import torch.nn as nn
torch.use_pos_weights = True

def _patch(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module._original_forward = module.forward
            module.forward = MethodType(pr_conv2d, module)
            print(module) 
        elif isinstance(module, nn.Linear):
            module._original_forward = module.forward
            module.forward = MethodType(pr_linear, module)
            print(module) 
    return model


def _recover(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
            # print(module) 
            module.forward = module._original_forward
        elif isinstance(module, nn.Linear) and hasattr(module, '_original_forward'):
            # print(module) 
            module.forward = module._original_forward
    return model
