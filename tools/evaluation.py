"""Perform re-evaluation on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time
from six.moves import cPickle as pickle
import torch

try:
    import _init_paths  # pylint: disable=unused-import
except:
    import tools._init_paths
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine import empty_results, extend_results
from core.test import box_results_for_corloc, box_results_with_nms_and_limit
from datasets.json_dataset import JsonDataset
from datasets import task_evaluation
import utils.logging
from datasets.json_inference import coco_inst_seg_eval
from utils.mask_eval_utils import coco_encode, mask_results_with_nms_and_limit, \
    mask_results_with_nms_and_limit_get_index
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import multiprocessing

from prm.coco_dataset import coco_nummap_id
import scipy
from prm.prm_configs import ismember

from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils.boxes import expand_boxes, clip_boxes_to_image
import numpy as np
# from scipy.misc import imresize
import json

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

from datasets.json_inference import coco_inst_seg_eval

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset', default='coco2017',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', default='configs/baselines/vgg16_cobcoco2017_semantic.yaml',  # required=True,
        help='optional config file')
    parser.add_argument(
        '--result_path', default='/mass/wsk/eccv2020_ppsn-master/Outputs/vgg16_cobcoco2014_em/BWSIS/v3/detections.pkl',
        help='the path for result file.')
    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results.')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')
    return parser.parse_args()

def eval(name, total, roidb):
    predictions = []
    range_begin = int(name) * len(roidb) // total
    if (int(name) + 1) == total:
        range_end = len(roidb)
    else:
        range_end = (int(name) + 1) * len(roidb) // total
    print('Process%s: range from %d to %d' % (name, range_begin, range_end))
    for i, entry in enumerate(roidb):
        if i < int(name) * len(roidb) // total:
            continue
        if i >= (int(name) + 1) * len(roidb) // total:
            continue
        boxes = all_boxes[entry['image']]
        scores = boxes['scores']
        boxes = boxes['boxes']

        if "coco" in args.dataset:
            cob_original_file = './data/coco2017/COB-COCO'
            file_n = entry['image'].split("/")[-1].replace(".jpg", ".mat")
            file_name = 'COCO_train2014_' + file_n
            if not os.path.exists(os.path.join(cob_original_file, file_name)):
                file_name = 'COCO_val2014_' + file_n
            if not os.path.exists(os.path.join(cob_original_file, file_name)):
                file_name = file_n
            mask_proposals = scipy.io.loadmat(
                os.path.join(cob_original_file, file_name),
                verify_compressed_data_integrity=False)['maskmat'].copy()
            cls_num = 81

        else:
            cob_original_file = './data/VOC2012/COB_SBD_val'
            file_name = entry['image'][-15:-4] + '.mat'
            mask_proposals = loadmat(os.path.join(cob_original_file, file_name))['maskmat'][:, 0]
            cls_num = 21

        if cfg.TEST.PROPOSAL_FILTER:
            image_area = entry['height'] * entry['width']
            invalid_index = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > proposal_size_limit[
                1] * image_area
            scores[invalid_index] = 0
            invalid_index = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) < proposal_size_limit[
                0] * image_area
            scores[invalid_index] = 0

        scores, boxes, cls_boxes, cls_inds = mask_results_with_nms_and_limit_get_index(cfg, scores, boxes)

        for cls_idx in range(1, cls_num):
            for instance_idx in range(len(cls_boxes[cls_idx])):  # instance_idx = 0
                COB_ind = cls_inds[cls_idx][instance_idx]
                mask = mask_proposals[COB_ind]
                if cls_num == 21:
                    predictions.append(dict(image_id=int(entry['id']),
                                            score=cls_boxes[cls_idx][instance_idx][4].astype(np.float64),
                                            category_id=int(cls_idx),
                                            segmentation=coco_encode(mask.astype(np.uint8))
                                            ))
                elif cls_num == 81:
                    predictions.append(dict(image_id=int(entry['id']),
                                            score=cls_boxes[cls_idx][instance_idx][4].astype(np.float64),
                                            category_id=coco_nummap_id[int(cls_idx) - 1],
                                            segmentation=coco_encode(mask.astype(np.uint8))
                                            ))
                else:
                    raise AssertionError
    with open(args.result_path[:-14] + 'sbd_instance_pred_origin' + '_' + str(range_begin) + '_' + str(range_end) +'.json', 'w') as f:
        f.write(json.dumps(predictions))

if __name__ == '__main__':
    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert os.path.exists(args.result_path), 'result_path doesnot exit'
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.result_path)
        logger.info('Automatically set output directory to %s', args.output_dir)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "coco2017val":
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 80
        annFile="./data/coco2017/annotations/instances_val2017.json"

    elif args.dataset == "coco2017test":
        cfg.TEST.DATASETS = ('coco_2017_test-dev',)
        cfg.MODEL.NUM_CLASSES = 80
        annFile = None # upload to website for evaluation

    elif args.dataset == 'voc2012sbdval':
        cfg.TEST.DATASETS = ('voc_2012_sbdval',)
        cfg.MODEL.NUM_CLASSES = 20
        annFile="./data/VOC2012/annotations/voc_2012_val.json"

    else:
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()

    logger.info('Re-evaluation with config:')

    with open(args.result_path, 'rb') as f:
        results = pickle.load(f)
        logger.info('Loading results from {}.'.format(args.result_path))
    all_boxes = results['all_boxes']

    dataset_name = cfg.TEST.DATASETS[0]
    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb()
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES + 1

    total_process = 12
    proposal_size_limit = (0.00002, 0.85)
    jobs = []
    for i in range(total_process):
        p = multiprocessing.Process(target=eval, args=(str(i), total_process, roidb))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()
    list_all = os.listdir(args.result_path[:-14])
    block = {}
    for name in list_all:
        if 'sbd_instance_pred_origin_' not in name:
            continue
        with open(os.path.join(args.result_path[:-14], name), 'rb') as f:
            block[name.split('.')[0]] = json.load(f)
            os.remove(os.path.join(args.result_path[:-14], name))

    predictions = []
    for i in range(len(block)):
        begin = i * len(roidb) // len(block)
        end = (i + 1) * len(roidb) // len(block)
        file_name = 'sbd_instance_pred_origin' + '_' + str(begin) + '_' + str(end)
        predictions.extend(block[file_name])

    with open(args.result_path[:-14] + 'sbd_instance_pred_origin.json', 'w') as f:
        f.write(json.dumps(predictions))

    result_file = args.result_path[:-14] + 'sbd_instance_pred_origin.json'

    if annFile == None:
        print("The json file needs to be uploaded to the website for evaluation")

    else:
        annType = ['segm', 'bbox', 'keypoints']
        annType = annType[0]
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(result_file)
        cocoEval = COCOeval(cocoGt, cocoDt, annType)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        mAP, cls_ap, cls_names = coco_inst_seg_eval(annFile, result_file)

        stack_item = []
        for key, value in cls_ap.items():
            stack_item.append(value)

        stack_item = np.concatenate(stack_item, axis=0).reshape((len(cls_ap.keys()),-1)).transpose()

        print('Class Performance(COCOAPI): ')

        for idx, _ in enumerate(cls_names):
            print("%-15s -->  %.1f, %.1f, %.1f, %.1f" % (cls_names[idx], 100 * stack_item[idx][0],
                                                        100 * stack_item[idx][1], 100 * stack_item[idx][2],
                                                        100 * stack_item[idx][3]))

        print('Performance(COCOAPI): ')
        for k, v in mAP.items():
            print('mAP@%s: %.1f' % (k, 100 * v))

    with open(result_file, 'r') as f:
        res = json.load(f)
        print("Summary len: {}".format(len(res)))