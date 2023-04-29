# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np

import cv2
import PIL.Image
import scipy.io
import matplotlib.pyplot as plt

from pycocotools import mask as COCOMask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def coco_encode(mask, width, height):
    mask = cv2.resize(mask.astype(np.uint8), dsize=(width, height),interpolation=cv2.INTER_NEAREST)
    mask[mask > 0] = 1
    encoding = COCOMask.encode(np.asfortranarray(mask))
    encoding['counts'] = encoding['counts'].decode('utf-8')
    return encoding

class InstanceEvaluator(object):
    def __init__(self, dataset_json, preds_json):
        self.dataset = COCO(dataset_json)
        self.object_classes = [v['name'] for v in self.dataset.loadCats(self.dataset.getCatIds())]
        self.preds = self.dataset.loadRes(preds_json)
        self.coco_eval = COCOeval(self.dataset, self.preds, 'segm')
        self.coco_eval.params.iouThrs = np.asarray([0.25, 0.5, 0.7, 0.75])

    def evaluate(self):
        mAP = dict()
        my_cls_ap = dict()
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()
        for thr_ind, thr in enumerate(self.coco_eval.params.iouThrs):
            ap_by_class = []
            for cls_ind, cls_name in enumerate(self.object_classes):
                cls_precision = self.coco_eval.eval['precision'][thr_ind, :, cls_ind, 0, -1]
                # cls_ap = np.mean(cls_precision[cls_precision > -1])
                tmp = cls_precision[cls_precision > -1]
                if len(tmp) != 0:
                     cls_ap = np.mean(tmp)
                else:
                     cls_ap = 0
                ap_by_class.append(cls_ap)
            mAP['%.2f' % thr] = np.asarray(ap_by_class).mean()
            my_cls_ap['%.2f' % thr] = ap_by_class
        return mAP, my_cls_ap, self.object_classes

def coco_inst_seg_eval(gt_file, pred_file):
    evaluator = InstanceEvaluator(gt_file, pred_file)
    return evaluator.evaluate()

if __name__ == '__main__':
    label_file =  '/mnt/jaring/cocoapi/cocoapi-master/MatlabAPI/results/*.json'
    result_file = '/mnt/yu/ynet/result/wsis/sbd/*.json'
    ########## 1. prm eval
    mAP, cls_ap = coco_inst_seg_eval( label_file, result_file)
    print('Performance(COCOAPI): ')
    for k, v in mAP.items():
        print('mAP@%s: %.1f' % (k, 100 * v))
    # print(cls_ap['0.50'])
    
    ########## 2. coco eval
    annType = ['segm','bbox','keypoints']
    annType = annType[0] 
    cocoGt=COCO(label_file)
    cocoDt=cocoGt.loadRes(result_file)
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    # imgIds=imgIds[0:100]
    # imgId = imgIds[np.random.randint(100)]
    # cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
