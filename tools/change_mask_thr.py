import json
import os

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
import cv2
import json
import numpy as np
from skimage import measure
from tqdm import tqdm
from PIL import Image
from pycocotools.cocoeval import COCOeval
import argparse

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def get_image_label(coco, img_id):
    annIds = coco.getAnnIds(img_id, iscrowd=0)
    anns = coco.loadAnns(annIds)
    cls_set = set([ann["category_id"] for ann in anns])

    return cls_set

# give annotation file, result file and mAP threshold
# return mAP score, like mAP75 and so on
# note: threshold is numpy.array type
def eval_ins_seg(annotation_file_path, result_file_path, thresholds):
    coco_gt = COCO(annotation_file_path)
    with open(result_file_path, 'r') as f:
        res = json.load(f)
        try:
            if "annotations" in res.keys():
                temp_filename = os.path.join("/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/data/trash",
                                             './temp.json')

                with open(temp_filename, 'w') as file_obj:
                    json.dump(res['annotations'], file_obj)

                result_file_path = temp_filename
                print(len(res['annotations']))
        except:
            temp_filename = os.path.join("/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/data/trash",
                                         './temp.json')

            with open(temp_filename, 'w') as file_obj:
                json.dump(res, file_obj)

            result_file_path = temp_filename

            print(len(res))

    coco_dt = coco_gt.loadRes(result_file_path)

    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.params.iouThrs = thresholds # set iou threshold

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    object_classes = [v['name'] for v in coco_gt.loadCats(coco_gt.getCatIds())]

    mAP = dict()
    cls_AP = dict()
    # eval["precision"] --> np.array(T,R,K,A,M)
    #                       T = len(p.iouThrs)
    #                       R = len(p.recThrs)
    #                       K = len(p.catIds) if p.useCats else 1
    #                       A = len(p.areaRng) --> select the size of object, all? small? mid? large?
    #                       M = len(p.maxDets) --> max det nums, 1, 10, 100?
    for thr_ind, thr in enumerate(coco_eval.params.iouThrs):
        ap_by_class = []
        for cls_ind, cls_name in enumerate(object_classes):
            cls_precision = coco_eval.eval['precision'][thr_ind, :, cls_ind, 0, -1]
            # cls_ap = np.mean(cls_precision[cls_precision > -1])
            tmp = cls_precision[cls_precision > -1]
            if len(tmp) != 0:
                cls_ap = np.mean(tmp)
            else:
                cls_ap = 0
            ap_by_class.append(cls_ap)
        mAP['%.2f' % thr] = np.asarray(ap_by_class).mean()
        cls_AP['%.2f' % thr] = ap_by_class
    return mAP, cls_AP, object_classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--org_dir', type=str,)
    parser.add_argument('--thr', type=float,default=0)

    args = parser.parse_args()

    org_dir = args.org_dir
    thr = args.thr

    json1 = os.path.join(org_dir,'msrcnn_pseudo_label.json')

    # label_file = "/home/lzc/WSIS-Benchmark/dataset/coco2017/annotations/instances_train2017.json"
    # label_file = "/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/annotations/voc_2012_trainaug.json"

    print(thr)
    filename = os.path.join(org_dir,'msrcnn_pseudo_label_{}.json'.format(thr))
    print(json1)
    print(filename)
    res = {}
    with open(json1, 'r', encoding='utf-8') as f:
        a = json.load(f)

    print("org len: " + str(len(a['annotations'])))

    res['annotations'] = []
    res['categories'] = a['categories']
    res['images'] = a['images']

    cls_id = {}
    j = 1
    for i in tqdm(range(len(a['annotations'])),total=len(a['annotations'])):
        if a['annotations'][i]['score'] < thr:
            pass

        else:
            a['annotations'][i]['id'] = j
            j += 1
            res['annotations'].append(a['annotations'][i])

    print("after thr len: " + str(len(res['annotations'])))
    with open(filename,'w') as file_obj:
        json.dump(res, file_obj)

    # mAP, cls_ap, class_name = eval_ins_seg(label_file, filename, np.asarray([0.25, 0.5, 0.7, 0.75]))
    #
    # stack_item = []
    # for key, value in cls_ap.items():
    #     stack_item.append(value)
    #
    # stack_item = np.concatenate(stack_item, axis=0).reshape((len(cls_ap.keys()),-1)).transpose()
    #
    # print('Class Performance(COCOAPI): ')
    #
    # for idx in range(len(class_name)):
    #     print("%-15s -->  %.1f, %.1f, %.1f, %.1f" % (class_name[idx], 100 * stack_item[idx][0],
    #                                         100 * stack_item[idx][1], 100 * stack_item[idx][2], 100 * stack_item[idx][3]))
    #
    #
    # print('Performance(COCOAPI): ')
    # for k, v in mAP.items():
    #     print('mAP@%s: %.1f' % (k, 100 * v))
    #
    #
