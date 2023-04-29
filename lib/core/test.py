# Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import numpy as np
import pycocotools.mask as mask_util

from torch.autograd import Variable
import torch

from core.config import cfg
from utils.timer import Timer
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.image as image_utils


def im_detect_all(model, im, box_proposals=None, masks = None, mat=None, timers=None,path=None,flag = None,labels=None):
    """Process the outputs of model for testing
    Args:
      model: the network module
      im_data: Pytorch variable. Input batch to the model.
      im_info: Pytorch variable. Input batch to the model.
      gt_boxes: Pytorch variable. Input batch to the model.
      num_boxes: Pytorch variable. Input batch to the model.
      args: arguments from command line.
      timer: record the cost of time for different steps
    The rest of inputs are of type pytorch Variables and either input to or output from the model.
    """
    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox'].tic()
    if cfg.TEST.BBOX_AUG.ENABLED:  # True
        scores, boxes, im_scale, blob_conv = im_detect_bbox_aug(
            model, im, box_proposals, masks, mat,path=path,flag=flag,labels =labels)
    else:
        scores, boxes, im_scale, blob_conv = im_detect_bbox(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, box_proposals, masks, mat,path=path,flag=flag,labels = labels )
    timers['im_detect_bbox'].toc()

    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class) (numpy.ndarray)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    # timers['misc_bbox'].tic()
    # scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
    # timers['misc_bbox'].toc()

    return {'scores': scores, 'boxes' : boxes}


def im_detect_bbox(model, im, target_scale, target_max_size, boxes=None, masks = None, mat=None, path=None,flag=None,labels = None ):
    """Prepare the bbox for testing"""

    inputs, im_scale = _get_blobs(im, boxes, mat, target_scale, target_max_size,flag)
    inputs['masks'] = masks
    inputs['path'] = [path]
    inputs['labels'] = labels

    if cfg.DEDUP_BOXES > 0:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        # [1,1019,5]
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        # [1,1019]
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]
        inputs['masks'] = masks[index, :]
        inputs['mat'] = np.array([0])
    else:
        index = np.arange(masks.shape[0])

    inputs['index'] = index
    # cfg.DEDUP_BOXES = 0
    inputs['mat'] = np.array([0])
    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['index'] = [Variable(torch.from_numpy(inputs['index']), volatile=True)]
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
        inputs['rois'] = [Variable(torch.from_numpy(inputs['rois']), volatile=True)]
        inputs['masks'] = [Variable(torch.from_numpy(inputs['masks']), volatile=True)]
        inputs['labels'] = [Variable(torch.from_numpy(inputs['labels']), volatile=True)]
        inputs['gtrois'] = [Variable(torch.from_numpy(np.empty((1, 5), dtype=np.float32)), volatile=True)]
        inputs['mat'] = [Variable(torch.from_numpy( inputs['mat'] ), volatile=True)]
    else:
        inputs['index'] = [torch.from_numpy(inputs['index'])]
        inputs['data'] = [torch.from_numpy(inputs['data'])]
        inputs['rois'] = [torch.from_numpy(inputs['rois'])]
        inputs['masks'] = [torch.from_numpy(inputs['masks'].astype(np.float32))]
        inputs['labels'] = [torch.from_numpy(inputs['labels'])]
        inputs['gtrois'] = [torch.from_numpy(np.empty((1, 5), dtype=np.float32))]
        inputs['mat'] = [torch.from_numpy( inputs['mat'] )  ]

    # print(inputs)
    return_dict = model(**inputs)

    # cls prob (activations after softmax)
    scores = return_dict['refine_score'][0].data.cpu().numpy().squeeze()
    for i in range(1, cfg.REFINE_TIMES):  # 
        scores += return_dict['refine_score'][i].data.cpu().numpy().squeeze()
    scores /= cfg.REFINE_TIMES
    # print(scores)
    # [1019, 21]
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])
    # [21,1019]
    pred_boxes = boxes

    if cfg.DEDUP_BOXES > 0:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, im_scale, return_dict['blob_conv']


def im_detect_bbox_aug(model, im, box_proposals=None, masks=None, mat = None,path=None,flag=None,labels = None):
    """Performs bbox detection with test-time augmentations.
    Function signature is the same as for im_detect_bbox.
    """
    assert not cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'
    assert not cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION', \
        'Coord heuristic must be union whenever score heuristic is union'
    assert not cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Score heuristic must be union whenever coord heuristic is union'

    # Collect detections computed under different transformations
    scores_ts = []
    boxes_ts = []

    def add_preds_t(scores_t, boxes_t):
        scores_ts.append(scores_t)
        boxes_ts.append(boxes_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:  # ture
        scores_hf, boxes_hf, _ = im_detect_bbox_hflip(
            model,
            im,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals, masks = masks, mat = mat,path=path,flag=flag,labels = labels
        )
        add_preds_t(scores_hf, boxes_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES: # (576, 688, 864, 1200)
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        scores_scl, boxes_scl = im_detect_bbox_scale(
            model, im, scale, max_size, box_proposals, masks,mat,path=path,flag=flag,labels = labels
        )
        add_preds_t(scores_scl, boxes_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            scores_scl_hf, boxes_scl_hf = im_detect_bbox_scale(
                model, im, scale, max_size, box_proposals, masks, mat, hflip=True,path=path,flag=flag,labels = labels
            )
            add_preds_t(scores_scl_hf, boxes_scl_hf)

    # Perform detection at different aspect ratios
    for aspect_ratio in cfg.TEST.BBOX_AUG.ASPECT_RATIOS: # ()
        scores_ar, boxes_ar = im_detect_bbox_aspect_ratio(
            model, im, aspect_ratio, box_proposals,flag=flag,labels = labels
        )
        add_preds_t(scores_ar, boxes_ar)

        if cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP:
            scores_ar_hf, boxes_ar_hf = im_detect_bbox_aspect_ratio(
                model, im, aspect_ratio, box_proposals, hflip=True,flag=flag,labels = labels
            )
            add_preds_t(scores_ar_hf, boxes_ar_hf)

    # Compute detections for the original image (identity transform) last to
    # ensure that the Caffe2 workspace is populated with blobs corresponding
    # to the original image on return (postcondition of im_detect_bbox)
    scores_i, boxes_i, im_scale_i, blob_conv_i = im_detect_bbox(
        model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals, masks = masks, mat = mat,path=path,flag=flag,labels = labels   # cfg.TEST.SCALE = 480, cfg.TEST.MAX_SIZE = 2000
    )
    add_preds_t(scores_i, boxes_i)

    # Combine the predicted scores
    if cfg.TEST.BBOX_AUG.SCORE_HEUR == 'ID':
        scores_c = scores_i
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'AVG':
        scores_c = np.mean(scores_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION':
        scores_c = np.vstack(scores_ts)
    else:
        raise NotImplementedError(
            'Score heur {} not supported'.format(cfg.TEST.BBOX_AUG.SCORE_HEUR)
        )

    # Combine the predicted boxes
    if cfg.TEST.BBOX_AUG.COORD_HEUR == 'ID':
        boxes_c = boxes_i
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'AVG':
        boxes_c = np.mean(boxes_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION':
        boxes_c = np.vstack(boxes_ts)
    else:
        raise NotImplementedError(
            'Coord heur {} not supported'.format(cfg.TEST.BBOX_AUG.COORD_HEUR)
        )

    return scores_c, boxes_c, im_scale_i, blob_conv_i


def im_detect_bbox_hflip(
        model, im, target_scale, target_max_size, box_proposals=None, masks=None, mat =None,path=None,flag=None,labels =None):
    """Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    # Compute predictions on the flipped image
    im_hf = im[:, ::-1, :]
    im_width = im.shape[1]

    box_proposals_hf = box_utils.flip_boxes(box_proposals, im_width)
    mat_hf = mat.copy()
    #mat_hf[:,:4] = box_utils.flip_boxes(mat_hf[:,:4], im_width)
    
    masks_hf =np.flip(masks.copy(),2)

    scores_hf, boxes_hf, im_scale, _ = im_detect_bbox(
        model, im_hf, target_scale, target_max_size, boxes=box_proposals_hf, masks = masks_hf, mat =  mat_hf,path=path,flag=flag,labels = labels
    )

    # Invert the detections computed on the flipped image
    boxes_inv = box_utils.flip_boxes(boxes_hf, im_width)

    return scores_hf, boxes_inv, im_scale


def im_detect_bbox_scale(
        model, im, target_scale, target_max_size, box_proposals=None, masks = None,mat =None, hflip=False,path=None,flag=None,labels=None):
    """Computes bbox detections at the given scale.
    Returns predictions in the original image space.
    """
    if hflip:
        scores_scl, boxes_scl, _ = im_detect_bbox_hflip(
            model, im, target_scale, target_max_size, box_proposals=box_proposals, masks=masks,mat =mat,path=path,flag=flag,labels = labels
        )
    else:
        scores_scl, boxes_scl, _, _ = im_detect_bbox(
            model, im, target_scale, target_max_size, boxes=box_proposals, masks=masks,mat =mat,path=path,flag=flag,labels = labels
        )
    return scores_scl, boxes_scl


def im_detect_bbox_aspect_ratio(
        model, im, aspect_ratio, box_proposals=None, hflip=False,path=None,flag=None,labels = None):
    """Computes bbox detections at the given width-relative aspect ratio.
    Returns predictions in the original image space.
    """
    # Compute predictions on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)

    box_proposals_ar = box_utils.aspect_ratio(box_proposals, aspect_ratio)

    if hflip:
        scores_ar, boxes_ar, _ = im_detect_bbox_hflip(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals_ar,
            path = path,flag=flag,labels = labels
        )
    else:
        scores_ar, boxes_ar, _, _ = im_detect_bbox(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            boxes=box_proposals_ar,
            path=path,flag=flag,
            labels = labels
        )

    # Invert the detected boxes
    boxes_inv = box_utils.aspect_ratio(boxes_ar, 1.0 / aspect_ratio)

    return scores_ar, boxes_inv


def box_results_for_corloc(scores, boxes):  # NOTE: support single-batch
    """Returns bounding-box detection results for CorLoc evaluation.

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES #####  +1 
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(num_classes ):   # 1, num_classes):   #
        max_ind = np.argmax(scores[:, j])
        cls_boxes[j] = np.hstack((boxes[max_ind, :].reshape(1, -1),
                               np.array([[scores[max_ind, j]]])))

######################
    new_box =  [[] for _ in range(num_classes+1)]
    for i in range(num_classes):
        new_box[i+1] = cls_boxes[i]
    cls_boxes = new_box
######################    

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes


def box_results_with_nms_and_limit(scores, boxes):  # NOTE: support single-batch
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES  #####  +1 
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range( num_classes ):   # 1, num_classes):   #
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0] # 1e-05
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, :]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
        if cfg.TEST.SOFT_NMS.ENABLED:    # False
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils.nms(dets_j, cfg.TEST.NMS) # 0.3
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:   # False
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:  # 100
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(num_classes)]  # 1, num_classes)]       # 
        )
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range( num_classes):  # 1, num_classes):  #
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

######################
    new_box =  [[] for _ in range(num_classes+1)]
    for i in range(num_classes):
        new_box[i+1] = cls_boxes[i]
    cls_boxes = new_box
######################    
    #print(num_classes)
    #print(len(cls_boxes))
    
    im_results = np.vstack([cls_boxes[j] for j in range( 1, num_classes)])
    
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes


def _get_rois_blob(im_rois, im_scale):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels


def _project_im_peakmat(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois =  im_rois
    rois[:,:4] = rois[:,:4]* im_scale_factor
    return rois

def _get_blobs(im, rois, mat, target_scale, target_max_size,flag):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale = \
        blob_utils.get_image_blob(im, target_scale, target_max_size, flag)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale)
    blobs['mat'] = mat # _project_im_peakmat(mat, im_scale)
    blobs['labels'] = np.zeros((1, cfg.MODEL.NUM_CLASSES), dtype=np.int32)
    return blobs, im_scale
