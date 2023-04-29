# -*- coding: utf-8 -*-

import utils.boxes as box_utils
import numpy as np

def mask_results_with_nms_and_limit(cfg, scores, boxes, masks):  # NOTE: support single-batch
    assert len(boxes) == len(masks)
    num_classes = cfg.MODEL.NUM_CLASSES  #####  +1 
    cls_boxes = [[] for _ in range(num_classes)]
    cls_masks = [[] for _ in range(num_classes)]
    for j in range( num_classes ):   # 1, num_classes):   #
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0] # 1e-05
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, :]
        masks_j = masks[inds, :]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)

        keep = box_utils.nms(dets_j, cfg.TEST.NMS) # 0.3
        nms_dets = dets_j[keep, :]
        cls_boxes[j] = nms_dets
        
        nms_dets_masks = masks_j[keep, :]
        cls_masks[j] = nms_dets_masks

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
                cls_masks[j] = cls_masks[j][keep, :]
                
######################
    new_box =  [[] for _ in range(num_classes+1)]
    for i in range(num_classes):
        new_box[i+1] = cls_boxes[i]
    cls_boxes = new_box
    
    new_mask =  [[] for _ in range(num_classes+1)]
    for i in range(num_classes):
        new_mask[i+1] = cls_masks[i]
    cls_masks = new_mask
######################    

    im_results = np.vstack([cls_boxes[j] for j in range( 1, num_classes)])
    
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes, cls_masks



def mask_results_with_nms_and_limit_get_index(cfg, scores, boxes, DETECTIONS_PER_IM = 100):  # NOTE: support single-batch

    num_classes = cfg.MODEL.NUM_CLASSES  #####  +1 
    cls_boxes = [[] for _ in range(num_classes)]
    cls_inds = [[] for _ in range(num_classes)]

    for j in range( num_classes ): 
        
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0] 
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, :]

        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)

        keep = box_utils.nms(dets_j, cfg.TEST.NMS) # 0.3
        nms_dets = dets_j[keep, :]
        cls_boxes[j] = nms_dets

        keep_index = np.array(range(len(scores)))
        keep_index_j = keep_index[inds]
        nms_index = keep_index_j[keep]
        cls_inds[j] = nms_index

    # Limit to max_per_image detections **over all classes**
    # if cfg.TEST.DETECTIONS_PER_IM > 0:  # 100
    if DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(num_classes)]  # 1, num_classes)]       # 
        )
        # print(len(image_scores))
        if len(image_scores) > DETECTIONS_PER_IM: # cfg.TEST.
            image_thresh = np.sort(image_scores)[-DETECTIONS_PER_IM] # cfg.TEST.
            for j in range( num_classes):  # 1, num_classes):  #
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]
                cls_inds[j] = cls_inds[j][keep]

    ######################
    new_box =  [[] for _ in range(num_classes+1)]
    for i in range(num_classes):
        new_box[i+1] = cls_boxes[i]
    cls_boxes = new_box
    
    new_inds =  [[] for _ in range(num_classes+1)]
    for i in range(num_classes):
        new_inds[i+1] = cls_inds[i]
    cls_inds = new_inds
    ######################    

    im_results = np.vstack([cls_boxes[j] for j in range( 1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]

    return scores, boxes, cls_boxes, cls_inds


from pycocotools import mask as COCOMask
def coco_encode(mask):
    encoding = COCOMask.encode(np.asfortranarray(mask) )
    encoding['counts'] = encoding['counts'].decode('utf-8')
    return encoding