# -*- coding: utf-8 -*-

from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils.boxes as box_utils
from core.config import cfg

import numpy as np
from sklearn.cluster import KMeans

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# boxes:      (1019, 5),  boxes
# mil_score:  [1019, 20], cls_prob
# im_labels:  [1,20],     im_labels
# refine:    [1019, 21]   cls_prob_new
'''   
cls_prob = mil_score
cls_prob_new = refine
''' 
def PCL(boxes, mat, im_labels):
    
    mat = mat.data.cpu().numpy()
    # proposals = _get_graph_centers(boxes.copy(), cls_prob.copy(),im_labels.copy())
    # print(proposals) {'gt_boxes': , 'gt_classes': , 'gt_scores'  }
    proposals = _get_prm_centers(boxes.copy(), mat.copy(), im_labels.copy())
    
    gt_assignment = _get_peak_proposal_clusters(boxes.copy(),
            proposals, im_labels.copy())

    return gt_assignment
'''
labels = labels.reshape(1, -1).astype(np.float32).copy()

'''


def _get_top_ranking_propoals(probs):
    """Get top ranking proposals by k-means"""
    kmeans = KMeans(n_clusters=cfg.TRAIN.NUM_KMEANS_CLUSTER, # 3
        random_state=cfg.RNG_SEED).fit(probs)
    high_score_label = np.argmax(kmeans.cluster_centers_)

    index = np.where(kmeans.labels_ == high_score_label)[0]

    if len(index) == 0:
        index = np.array([np.argmax(probs)])

    return index

def _build_graph(boxes, iou_threshold):
    """Build graph based on box IoU"""
    overlaps = box_utils.bbox_overlaps(
        boxes.astype(dtype=np.float32, copy=False),
        boxes.astype(dtype=np.float32, copy=False))
    # print(overlaps.shape) (112, 112) (344, 344)  (87, 87)
    return (overlaps > iou_threshold).astype(np.float32)



def _get_prm_centers(boxes, mat, im_labels):
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :].copy() # im_labels_tmp [20]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for c_idx in range(len(mat)):
        overlaps = box_utils.bbox_overlaps( boxes.astype(dtype=np.float32, copy=False),
             mat[:,:4].astype(dtype=np.float32, copy=False))# (200, 5)
        index = np.where(overlaps[:,c_idx]==overlaps[:,c_idx].max())[0] # overlaps[index,c_idx]
        if len(index)!=1:
            index = index[0]
        gt_boxes_tmp  = boxes[index].copy()
        gt_classes_tmp = mat[c_idx,4].astype(np.int)
        gt_scores_tmp = np.array([1])
        
        gt_boxes = np.vstack((gt_boxes, gt_boxes_tmp))
        gt_scores = np.vstack((gt_scores,gt_scores_tmp.reshape(-1, 1)))
        gt_classes = np.vstack( ( gt_classes, gt_classes_tmp.reshape(-1, 1) + 1 ))
        boxes = np.delete(boxes.copy(), index , axis=0)
    assert len(gt_boxes) == len(gt_classes), 'equal length'
    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}
    return proposals


# (1019, 5),   all_rois
# proposals:      gt_boxes (5, 4)        gt_labels (5, 1)       gt_scores (5, 1)
# [1,20],      im_labels
# [1019, 21]   cls_prob_new
'''
all_rois = boxes.copy()
proposals = proposals
im_labels = im_labels.copy()
cls_prob = cls_prob_new.copy()
'''    
def _get_peak_proposal_clusters(all_rois, proposals, im_labels):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # print(all_rois.shape) # (1019, 4)
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    # print(gt_boxes.shape)  # (5, 4)
    # print(gt_labels.shape) # (5, 1)
    # print(gt_scores.shape) # (5, 1)
    overlaps = box_utils.bbox_overlaps(
        all_rois.astype(dtype=np.float32, copy=False),
        gt_boxes.astype(dtype=np.float32, copy=False))
    # print(overlaps.shape) # (1019, 5)
    gt_assignment = overlaps.argmax(axis=1)
    gt_assignment = gt_labels[gt_assignment, 0]
    
    # print(gt_assignment.shape) # (1019,)
    try:
        max_overlaps = overlaps.max(axis=1)
    except:
        print('gtboxes=',gt_boxes)
        print('gtlabels=',gt_labels)
        print(gt_labels.shape)
        print(gt_assignment.shape)
        print('-'*20)

    # Select background RoIs as those with < FG_THRESH overlap
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]
    gt_assignment[bg_inds] = -1

    return gt_assignment


def _get_proposal_clusters(all_rois, proposals, im_labels, cls_prob):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # print(all_rois.shape) # (1019, 4)
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    # print(gt_boxes.shape)  # (5, 4)
    # print(gt_labels.shape) # (5, 1)
    # print(gt_scores.shape) # (5, 1)
    overlaps = box_utils.bbox_overlaps(
        all_rois.astype(dtype=np.float32, copy=False),
        gt_boxes.astype(dtype=np.float32, copy=False))
    # print(overlaps.shape) # (1019, 5)
    gt_assignment = overlaps.argmax(axis=1)

    # print(gt_assignment.shape) # (1019,)
    try:
        max_overlaps = overlaps.max(axis=1)
        labels = gt_labels[gt_assignment, 0]
    except:
        print('gtboxes=',gt_boxes)
        print('gtlabels=',gt_labels)
        print(gt_labels.shape)
        print(gt_assignment.shape)
        print('-'*20)
    
    cls_loss_weights = gt_scores[gt_assignment, 0]
 
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0] # 0.5

    # Select background RoIs as those with < FG_THRESH overlap
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]

    ig_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH)[0]
    cls_loss_weights[ig_inds] = 0.0

    labels[bg_inds] = 0
    gt_assignment[bg_inds] = -1

    img_cls_loss_weights = np.zeros(gt_boxes.shape[0], dtype=np.float32) # len(gt_boxes)
    pc_probs = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_labels = np.zeros(gt_boxes.shape[0], dtype=np.int32)
    pc_count = np.zeros(gt_boxes.shape[0], dtype=np.int32)

    for i in xrange(gt_boxes.shape[0]):
        po_index = np.where(gt_assignment == i)[0]
        img_cls_loss_weights[i] = np.sum(cls_loss_weights[po_index])
        pc_labels[i] = gt_labels[i, 0]
        pc_count[i] = len(po_index)
        pc_probs[i] = np.average(cls_prob[po_index, pc_labels[i]])
    
    return labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, pc_count, img_cls_loss_weights


class PCLLosses(torch.autograd.Function):

    def forward(ctx, pcl_probs, labels, cls_loss_weights,
                gt_assignment, pc_labels, pc_probs, pc_count,
                img_cls_loss_weights, im_labels):
        ctx.pcl_probs, ctx.labels, ctx.cls_loss_weights, \
        ctx.gt_assignment, ctx.pc_labels, ctx.pc_probs, \
        ctx.pc_count, ctx.img_cls_loss_weights, ctx.im_labels = \
        pcl_probs, labels, cls_loss_weights, gt_assignment, \
        pc_labels, pc_probs, pc_count, img_cls_loss_weights, im_labels

        batch_size, channels = pcl_probs.size()
        loss = 0
        ctx.mark_non_differentiable(labels, cls_loss_weights,
                                    gt_assignment, pc_labels, pc_probs,
                                    pc_count, img_cls_loss_weights, im_labels)

        for c in range(channels):
            if im_labels[0, c] != 0:
                if c == 0: # 0: backgroud
                    for i in range(batch_size):
                        if labels[0, i] == 0:
                            loss -= cls_loss_weights[0, i] * torch.log(pcl_probs[i, c])
                else:
                    for i in range(pc_labels.size(0)):
                        if pc_probs[0, i] == c:
                            loss -= img_cls_loss_weights[0, i] * torch.log(pc_probs[0, i])

        return loss / batch_size

    def backward(ctx, grad_output):
        pcl_probs, labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, \
        pc_count, img_cls_loss_weights, im_labels = \
        ctx.pcl_probs, ctx.labels, ctx.cls_loss_weights, \
        ctx.gt_assignment, ctx.pc_labels, ctx.pc_probs, \
        ctx.pc_count, ctx.img_cls_loss_weights, ctx.im_labels

        grad_input = grad_output.new(pcl_probs.size()).zero_()

        batch_size, channels = pcl_probs.size()

        for i in range(batch_size):
            for c in range(channels):
                grad_input[i, c] = 0
                if im_labels[0, c] != 0:
                    if c == 0:
                        if labels[0, i] == 0:
                            grad_input[i, c] = -cls_loss_weights[0, i] / pcl_probs[i, c]
                    else:
                        if labels[0, i] == c:
                            pc_index = int(gt_assignment[0, i].item())
                            if c != pc_labels[0, pc_index]:
                                print('labels mismatch.')
                            grad_input[i, c] = -img_cls_loss_weights[0, pc_index] / (pc_count[0, pc_index] * pc_probs[0, pc_index])

        grad_input /= batch_size

        return grad_input, grad_output.new(labels.size()).zero_(), grad_output.new(cls_loss_weights.size()).zero_(), \
               grad_output.new(gt_assignment.size()).zero_(), grad_output.new(pc_labels.size()).zero_(), \
               grad_output.new(pc_probs.size()).zero_(), grad_output.new(pc_count.size()).zero_(), \
               grad_output.new(img_cls_loss_weights.size()).zero_(), grad_output.new(im_labels.size()).zero_()