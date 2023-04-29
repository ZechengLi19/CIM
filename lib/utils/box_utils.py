# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def find_bbox(mask):
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    stats = stats[stats[:,4].argsort()]
    return stats[:-1]

def find_bbox_with_outlier(mask):
    if mask.all() == False:
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        stats = stats[stats[:,4].argsort()]
        stats = stats[:-1][0]
        ymin,xmin,ymax,xmax = stats[1], stats[0], stats[1]+stats[3], stats[0]+stats[2]
    else:
        xmin, ymin, xmax, ymax = 0,0,mask.shape[1]-1,mask.shape[0]-1
    return xmin, ymin, xmax, ymax


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (np.array[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (np.array[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
def box_overlap( boxes1, boxes2 ):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (np.array[N, 4])
        boxes2 (np.array[M, 4])

    Returns:
        iou (np.array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = np.maximum((rb - lt), 0) # (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
