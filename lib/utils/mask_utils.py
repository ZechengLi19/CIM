# -*- coding: utf-8 -*-

from chainer.backends import cuda


def mask_iou(mask_a, mask_b):
    if mask_a.shape[1:] != mask_b.shape[1:]:
        raise IndexError
    xp = cuda.get_array_module(mask_a)
    n_mask_a = len(mask_a)
    n_mask_b = len(mask_b)
    iou = xp.empty((n_mask_a, n_mask_b), dtype=xp.float32)
    for n, m_a in enumerate(mask_a):
        for k, m_b in enumerate(mask_b):
            intersect = xp.bitwise_and(m_a, m_b).sum()
            union = xp.bitwise_or(m_a, m_b).sum()
            iou[n, k] = intersect / union
    return iou

def mask_asymmetric_iou(mask_a, mask_b):
    if mask_a.shape[1:] != mask_b.shape[1:]:
        raise IndexError
    xp = cuda.get_array_module(mask_a)
    n_mask_a = len(mask_a)
    n_mask_b = len(mask_b)
    iou = xp.empty((n_mask_a, n_mask_b), dtype=xp.float32)
    union = mask_b.sum()
    for n, m_a in enumerate(mask_a): # 200
        for k, m_b in enumerate(mask_b): # 1
            intersect = xp.bitwise_and(m_a, m_b).sum()
            iou[n, k] = intersect / union
    return iou


def mask_inside(mask_a, mask_b):
    if mask_a.shape[1:] != mask_b.shape[1:]:
        raise IndexError
    xp = cuda.get_array_module(mask_a)
    n_mask_a = len(mask_a)
    n_mask_b = len(mask_b)
    iou = xp.empty((n_mask_a, n_mask_b), dtype=xp.float32)
    for n, m_a in enumerate(mask_a):
        for k, m_b in enumerate(mask_b):
            intersect = xp.bitwise_and(m_a, m_b).sum()
            union = xp.bitwise_or(m_b, m_b).sum()
            iou[n, k] = intersect / union
    return iou


def mask_outside(mask_a, mask_b):
    if mask_a.shape[1:] != mask_b.shape[1:]:
        raise IndexError
    xp = cuda.get_array_module(mask_a)
    n_mask_a = len(mask_a)
    n_mask_b = len(mask_b)
    iou = xp.empty((n_mask_a, n_mask_b), dtype=xp.float32)
    for n, m_a in enumerate(mask_a):
        for k, m_b in enumerate(mask_b):
            intersect = xp.bitwise_and(m_a, m_b).sum()
            union = xp.bitwise_or(m_a, m_a).sum()
            iou[n, k] = intersect / union
    return iou