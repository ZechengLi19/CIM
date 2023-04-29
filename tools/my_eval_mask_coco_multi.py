# # -*- coding: utf-8 -*-
#
# # -*- coding: utf-8 -*-
#
# """Perform inference on one or more datasets."""
# """Perform re-evaluation on one or more datasets."""
#
# import argparse
# import cv2
# import os
# import pprint
# import sys
# import time
# from six.moves import cPickle as pickle
# import torch
#
# try:
#     import _init_paths  # pylint: disable=unused-import
# except:
#     import tools._init_paths
# from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
# from core.test_engine import empty_results, extend_results
# from core.test import box_results_for_corloc, box_results_with_nms_and_limit
# from datasets.json_dataset import JsonDataset
# from datasets import task_evaluation
# import utils.logging
# from datasets.json_inference import coco_inst_seg_eval
# from utils.mask_eval_utils import coco_encode, mask_results_with_nms_and_limit, \
#     mask_results_with_nms_and_limit_get_index
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# import multiprocessing
#
# from prm.coco_dataset import coco_nummap_id
# import scipy
# from prm.prm_configs import ismember
#
# from scipy.io import loadmat
# import matplotlib.pyplot as plt
# from utils.boxes import expand_boxes, clip_boxes_to_image
# import numpy as np
# # from scipy.misc import imresize
# import json
#
# # OpenCL may be enabled by default in OpenCV3; disable it because it's not
# # thread safe and causes unwanted GPU memory allocations.
# cv2.ocl.setUseOpenCL(False)
#
#
# def parse_args():
#     """Parse in command line arguments"""
#     parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
#     parser.add_argument(
#         '--dataset', default='coco2017',
#         help='training dataset')
#     parser.add_argument(
#         '--cfg', dest='cfg_file', default='configs/baselines/vgg16_cobcoco2014.yaml',  # required=True,
#         help='optional config file')
#     parser.add_argument(
#         '--result_path', default='/mass/wsk/eccv2020_ppsn-master/Outputs/vgg16_cobcoco2014_em/BWSIS/v3/detections.pkl',
#         help='the path for result file.')
#     parser.add_argument(
#         '--output_dir',
#         help='output directory to save the testing results.')
#     parser.add_argument(
#         '--set', dest='set_cfgs',
#         help='set config keys, will overwrite config in the cfg_file.'
#              ' See lib/core/config.py for all options',
#         default=[], nargs='*')
#     return parser.parse_args()
#
# def eval(name, total, roidb):
#     predictions = []
#     range_begin = int(name) * len(roidb) // total
#     if (int(name) + 1) == total:
#         range_end = len(roidb)
#     else:
#         range_end = (int(name) + 1) * len(roidb) // total
#     print('Process%s: range from %d to %d' % (name, range_begin, range_end))
#     for i, entry in enumerate(roidb):
#         if i < int(name) * len(roidb) // total:
#             continue
#         if i >= (int(name) + 1) * len(roidb) // total:
#             continue
#         try:
#             boxes = all_boxes[entry['image']]
#         except:
#             boxes = all_boxes[
#                 entry['image'].replace('/mass/wsk/eccv2020_ppsn-master/', '/mnt/th/Pycharm_Projects/EM_wsk_refine/')]
#         scores = boxes['scores']  # [:,1:]
#         boxes = boxes['boxes']
#
#         # proposal
#         # COBmat = scipy.io.loadmat(
#         #         os.path.join('data/coco2014/COB-COCO-val2014-proposals', entry['image'][-29:-4] + '.mat'),    # str(entry['id'])
#         #         verify_compressed_data_integrity=False)
#         # COBlabels = COBmat['labels']
#         # COBsuperpixel = COBmat['superpixels']
#         # if args.dataset == "coco":
#         #     file_name = 'COCO_val2014_' + entry['image'][-16:-4] + '.mat'
#         #     if not os.path.exists(os.path.join('data/coco2014/COB-COCO', file_name)):
#         #         file_name = 'COCO_train2014_' + entry['image'][-16:-4] + '.mat'
#         #     COB_proposals = scipy.io.loadmat(
#         #         os.path.join('data/coco2014/COB-COCO', file_name),
#         #         verify_compressed_data_integrity=False)['maskmat']
#         #     mask_proposals = COB_proposals.copy()
#         #     num_proposal = len(mask_proposals)
#         # else:
#         #     mask_proposals = scipy.io.loadmat(
#         #         os.path.join('data/coco2014/COB-COCO', entry['image'][-29:-4] + '.mat'),
#         #         verify_compressed_data_integrity=False)['maskmat'].copy()
#         #     # mask_proposals = COBmat.copy()
#         #     num_proposal = len(mask_proposals)
#
#         if "coco" in args.dataset:
#             cob_original_file = '/home/lzc/WSIS-Benchmark/dataset/coco2017/COB-COCO'
#             file_n = entry['image'].split("/")[-1].replace(".jpg", ".mat")
#             file_name = 'COCO_train2014_' + file_n
#             if not os.path.exists(os.path.join(cob_original_file, file_name)):
#                 file_name = 'COCO_val2014_' + file_n
#             if not os.path.exists(os.path.join(cob_original_file, file_name)):
#                 file_name = file_n
#             mask_proposals = scipy.io.loadmat(
#                 os.path.join(cob_original_file, file_name),
#                 verify_compressed_data_integrity=False)['maskmat'].copy()
#             cls_num = 81
#
#
#         else:
#             file_name = entry['image'][-15:-4]
#             mask_proposals = loadmat(os.path.join('/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/output', file_name + '.mat'))[
#                 'maskmat'][:, 0]
#             cls_num = 21
#
#         # file_name = 'COCO_val2014_' + entry['image'][-16:-4] + '.mat'
#         # if not os.path.exists(os.path.join('data/coco2014/COB-COCO', file_name)):
#         #     file_name = 'COCO_train2014_' + entry['image'][-16:-4] + '.mat'
#         # COB_proposals = scipy.io.loadmat(
#         #     os.path.join('data/coco2014/COB-COCO', file_name),
#         #     verify_compressed_data_integrity=False)['maskmat']
#         # mask_proposals = COB_proposals.copy()
#
#         num_proposal = len(mask_proposals)
#
#         '''
#         proposals = []
#         for COB_ind in range( min(200, len(COBlabels)) ): # COB_ind = 0
#             proposals.append( ismember(np.array(COBsuperpixel), np.array(COBlabels[COB_ind][0][0]) ))
#         '''
#
#         if cfg.TEST.PROPOSAL_FILTER:
#             image_area = entry['height'] * entry['width']
#             invalid_index = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > proposal_size_limit[
#                 1] * image_area
#             scores[invalid_index] = 0
#             invalid_index = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) < proposal_size_limit[
#                 0] * image_area
#             scores[invalid_index] = 0
#
#         # scores, boxes, cls_boxes, cls_masks = mask_results_with_nms_and_limit(cfg, scores, boxes, np.stack(proposals))
#         scores, boxes, cls_boxes, cls_inds = mask_results_with_nms_and_limit_get_index(cfg, scores, boxes)
#
#         for cls_idx in range(1, cls_num):
#             for instance_idx in range(len(cls_boxes[cls_idx])):  # instance_idx = 0
#                 # print(cls_masks[cls_idx][instance_idx].shape)
#                 # print(cls_masks[cls_idx][instance_idx])
#                 # print(cls_masks[cls_idx][instance_idx].dtype)
#                 COB_ind = cls_inds[cls_idx][instance_idx]
#                 # mask = ismember(np.array(COBsuperpixel), np.array(COBlabels[COB_ind][0][0]) ).astype(np.uint8)
#                 mask = mask_proposals[COB_ind]
#                 predictions.append(dict(image_id=int(entry['id']),
#                                         score=cls_boxes[cls_idx][instance_idx][4].astype(np.float64),
#                                         category_id=coco_nummap_id[int(cls_idx) - 1],  # int(cls_idx),
#                                         segmentation=coco_encode(mask.astype(np.uint8))
#                                         ))
#         # extend_results(i, final_boxes, cls_box    es)
#         print(f'\rImage Index: {(i + 1):.0f}/{num_images:.0f}  ', end='')
#     with open(args.result_path[:-14] + 'sbd_instance_pred_origin' + '_' + str(range_begin) + '_' + str(range_end) +'.json', 'w') as f:
#         f.write(json.dumps(predictions))
#
# if __name__ == '__main__':
#
#     logger = utils.logging.setup_logging(__name__)
#     args = parse_args()
#     logger.info('Called with args:')
#     logger.info(args)
#
#     assert os.path.exists(args.result_path), 'result_path doesnot exit'
#     if args.output_dir is None:
#         args.output_dir = os.path.dirname(args.result_path)
#         logger.info('Automatically set output directory to %s', args.output_dir)
#     if args.cfg_file is not None:
#         merge_cfg_from_file(args.cfg_file)
#     if args.set_cfgs is not None:
#         merge_cfg_from_list(args.set_cfgs)
#     if args.cfg_file is not None:
#         merge_cfg_from_file(args.cfg_file)
#     if args.set_cfgs is not None:
#         merge_cfg_from_list(args.set_cfgs)
#
#     if args.dataset == "coco2017val":
#         cfg.TEST.DATASETS = ('coco_2017_val',)
#         cfg.MODEL.NUM_CLASSES = 80
#     elif args.dataset == "coco2017train":
#         cfg.TEST.DATASETS = ('coco_2017_train',)
#         cfg.MODEL.NUM_CLASSES = 80
#         cfg.TEST.PROPOSAL_FILES = cfg.TRAIN.PROPOSAL_FILES
#     elif args.dataset == 'voc2012sbdval':
#         cfg.TEST.DATASETS = ('voc_2012_sbdval',)
#         cfg.MODEL.NUM_CLASSES = 20
#     elif args.dataset == 'voc2012sbdval_style':
#         cfg.TEST.DATASETS = ('voc_2012_sbdval_style',)
#         cfg.MODEL.NUM_CLASSES = 20
#     elif args.dataset == 'voc2012trainaug':
#         cfg.TEST.DATASETS = ('voc_2012_trainaug',)
#         cfg.MODEL.NUM_CLASSES = 20
#         cfg.TEST.PROPOSAL_FILES = cfg.TRAIN.PROPOSAL_FILES
#     else:
#         assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
#     assert_and_infer_cfg()
#
#     logger.info('Re-evaluation with config:')
#     logger.info(pprint.pformat(cfg))
#
#     with open(args.result_path, 'rb') as f:
#         results = pickle.load(f)
#         logger.info('Loading results from {}.'.format(args.result_path))
#     all_boxes = results['all_boxes']
#
#     dataset_name = cfg.TEST.DATASETS[0]
#     dataset = JsonDataset(dataset_name)
#     roidb = dataset.get_roidb()
#     num_images = len(roidb)
#     num_classes = cfg.MODEL.NUM_CLASSES + 1
#     # final_boxes = empty_results(num_classes, num_images)
#
#     # %%
#
#     total_process = 8
#     proposal_size_limit = (0.00002, 0.85)
#     jobs = []
#     for i in range(total_process):
#         p = multiprocessing.Process(target=eval, args=(str(i), total_process, roidb))
#         jobs.append(p)
#         p.start()
#     for p in jobs:
#         p.join()
#     # list_all = os.listdir(args.result_path[:-14])
#     block = {}
#     # for name in list_all:
#     process = 0
#     while process != total_process:
#         range_begin = int(process) * len(roidb) // total_process
#         if (int(process) + 1) == total_process:
#             range_end = len(roidb)
#         else:
#             range_end = (int(process) + 1) * len(roidb) // total_process
#         try:
#             with open(os.path.join(args.result_path[:-14], 'sbd_instance_pred_origin' + '_' + str(range_begin) + '_' + str(range_end) +'.json'), 'rb') as f:
#                 block['sbd_instance_pred_origin' + '_' + str(range_begin) + '_' + str(range_end)] = json.load(f)
#                 os.remove(os.path.join(args.result_path[:-14], 'sbd_instance_pred_origin' + '_' + str(range_begin) + '_' + str(range_end)+'.json'))
#                 process += 1
#         except:
#             pass
#
#     assert len(block) == total_process
#     predictions = []
#     for i in range(len(block)):
#         begin = i * len(roidb) // len(block)
#         end = (i + 1) * len(roidb) // len(block)
#         file_name = 'sbd_instance_pred_origin' + '_' + str(begin) + '_' + str(end)
#         predictions.extend(block[file_name])
#
#
#     with open(args.result_path[:-14] + 'sbd_instance_pred_origin.json', 'w') as f:
#         f.write(json.dumps(predictions))
#
#
#     # results = task_evaluation.evaluate_all(dataset, final_boxes, args.output_dir, False)
#     # annFile = "data/coco2014/annotations/instances_val2017.json"
#     if "coco" in args.dataset:
#         annFile="/home/lzc/WSIS-Benchmark/dataset/coco2017/annotations/instances_val2017.json"
#     else:
#         annFile="/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/annotations/voc_2012_val.json"
#
#     result_file = args.result_path[:-14] + 'sbd_instance_pred_origin.json'
#
#     annType = ['segm', 'bbox', 'keypoints']
#     annType = annType[0]
#     cocoGt = COCO(annFile)
#     cocoDt = cocoGt.loadRes(result_file)
#     cocoEval = COCOeval(cocoGt, cocoDt, annType)
#     # cocoEval.params.imgIds  = imgIds
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()
#
#     mAP, cls_ap, cls_names = coco_inst_seg_eval(annFile, result_file)
#
#     stack_item = []
#     for key, value in cls_ap.items():
#         stack_item.append(value)
#
#     stack_item = np.concatenate(stack_item, axis=0).reshape((len(cls_ap.keys()),-1)).transpose()
#
#     print('Class Performance(COCOAPI): ')
#
#     for idx, _ in enumerate(cls_names):
#         print("%-10s -->  %.1f, %.1f, %.1f" % (cls_names[idx], 100 * stack_item[idx][0],
#                                             100 * stack_item[idx][1], 100 * stack_item[idx][2]))
#
#     print('Performance(COCOAPI): ')
#     for k, v in mAP.items():
#         print('mAP@%s: %.1f' % (k, 100 * v))
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

"""Perform inference on one or more datasets."""
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
        try:
            boxes = all_boxes[entry['image']]
        except:
            boxes = all_boxes[
                entry['image'].replace("/home/data2/Dataset/VOC2012/VOC2012_old/JPEGImages/",
                                        '/home/data2/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/data/VOC2012/JPEGImages/'
                                       )]
        scores = boxes['scores']  # [:,1:]
        boxes = boxes['boxes']

        # # proposal
        # # COBmat = scipy.io.loadmat(
        # #         os.path.join('data/coco2014/COB-COCO-val2014-proposals', entry['image'][-29:-4] + '.mat'),    # str(entry['id'])
        # #         verify_compressed_data_integrity=False)
        # # COBlabels = COBmat['labels']
        # # COBsuperpixel = COBmat['superpixels']
        # if args.dataset == "coco2017":
        #     file_name = 'COCO_val2014_' + entry['image'][-16:-4] + '.mat'
        #     if not os.path.exists(os.path.join('data/coco2014/COB-COCO', file_name)):
        #         file_name = 'COCO_train2014_' + entry['image'][-16:-4] + '.mat'
        #     COB_proposals = scipy.io.loadmat(
        #         os.path.join('data/coco2014/COB-COCO', file_name),
        #         verify_compressed_data_integrity=False)['maskmat']
        #     mask_proposals = COB_proposals.copy()
        #     num_proposal = len(mask_proposals)
        # else:
        #     mask_proposals = scipy.io.loadmat(
        #         os.path.join('data/coco2014/COB-COCO', entry['image'][-29:-4] + '.mat'),
        #         verify_compressed_data_integrity=False)['maskmat'].copy()
        #     # mask_proposals = COBmat.copy()
        #     num_proposal = len(mask_proposals)

        if "coco" in args.dataset:
            cob_original_file = '/home/lzc/WSIS-Benchmark/dataset/coco2017/COB-COCO'
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
            file_name = entry['image'][-15:-4]
            mask_proposals = loadmat(os.path.join('/home/data2/Dataset/VOC2012/VOC2012_old/output', file_name + '.mat'))['maskmat'][:, 0]
            cls_num = 21

        '''
        proposals = []
        for COB_ind in range( min(200, len(COBlabels)) ): # COB_ind = 0
            proposals.append( ismember(np.array(COBsuperpixel), np.array(COBlabels[COB_ind][0][0]) ))
        '''

        if cfg.TEST.PROPOSAL_FILTER:
            image_area = entry['height'] * entry['width']
            invalid_index = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > proposal_size_limit[
                1] * image_area
            scores[invalid_index] = 0
            invalid_index = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) < proposal_size_limit[
                0] * image_area
            scores[invalid_index] = 0

        # scores, boxes, cls_boxes, cls_masks = mask_results_with_nms_and_limit(cfg, scores, boxes, np.stack(proposals))
        scores, boxes, cls_boxes, cls_inds = mask_results_with_nms_and_limit_get_index(cfg, scores, boxes)

        for cls_idx in range(1, cls_num):
            for instance_idx in range(len(cls_boxes[cls_idx])):  # instance_idx = 0
                # print(cls_masks[cls_idx][instance_idx].shape)
                # print(cls_masks[cls_idx][instance_idx])
                # print(cls_masks[cls_idx][instance_idx].dtype)
                COB_ind = cls_inds[cls_idx][instance_idx]
                # mask = ismember(np.array(COBsuperpixel), np.array(COBlabels[COB_ind][0][0]) ).astype(np.uint8)
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
        # extend_results(i, final_boxes, cls_box    es)
        # print(f'\rImage Index: {(i + 1):.0f}/{num_images:.0f}  ', end='')
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
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "coco2017val":
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 80
        annFile="/home/lzc/WSIS-Benchmark/dataset/coco2017/annotations/instances_val2017.json"

    elif args.dataset == "coco2017test":
        cfg.TEST.DATASETS = ('coco_2017_test-dev',)
        cfg.MODEL.NUM_CLASSES = 80
        annFile = "/home/lzc/WSIS-Benchmark/dataset/coco2017/annotations/instances_val2017.json"

    elif args.dataset == "coco2017train":
        cfg.TEST.DATASETS = ('coco_2017_train',)
        cfg.MODEL.NUM_CLASSES = 80
        cfg.TEST.PROPOSAL_FILES = cfg.TRAIN.PROPOSAL_FILES
        annFile="/home/lzc/WSIS-Benchmark/dataset/coco2017/annotations/instances_train2017.json"

    elif args.dataset == 'voc2012sbdval':
        cfg.TEST.DATASETS = ('voc_2012_sbdval',)
        cfg.MODEL.NUM_CLASSES = 20
        annFile="/home/data2/Dataset/VOC2012/VOC2012_old/annotations/voc_2012_val.json"

    elif args.dataset == 'voc2012sbdval_style':
        cfg.TEST.DATASETS = ('voc_2012_sbdval_style',)
        cfg.MODEL.NUM_CLASSES = 20
    elif args.dataset == 'voc2012trainaug':
        cfg.TEST.DATASETS = ('voc_2012_trainaug',)
        cfg.MODEL.NUM_CLASSES = 20
        cfg.TEST.PROPOSAL_FILES = cfg.TRAIN.PROPOSAL_FILES
        train_json_file = "/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/annotations/voc_2012_trainaug.json"

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
    # final_boxes = empty_results(num_classes, num_images)

    # %%

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

    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[0]
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(result_file)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


    def eval_ins_seg(annotation_file_path, result_file_path, thresholds):
        coco_gt = COCO(annotation_file_path)
        coco_dt = coco_gt.loadRes(result_file_path)

        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval.params.iouThrs = thresholds  # set iou threshold
        print(thresholds)

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
        return mAP, cls_AP

    # mAP, cls_ap = eval_ins_seg(annFile, result_file,np.arange(0,1.05,0.1))
    mAP, cls_ap = eval_ins_seg(annFile, result_file,np.array([0.25, 0.5,0.7,0.75]))


    print(mAP)
    print(cls_ap)

    # stack_item = []
    # for key, value in cls_ap.items():
    #     stack_item.append(value)
    #
    # stack_item = np.concatenate(stack_item, axis=0).reshape((len(cls_ap.keys()),-1)).transpose()
    #
    # print('Class Performance(COCOAPI): ')
    #
    # for idx, _ in enumerate(cls_names):
    #     print("%-15s -->  %.1f, %.1f, %.1f, %.1f" % (cls_names[idx], 100 * stack_item[idx][0],
    #                                         100 * stack_item[idx][1], 100 * stack_item[idx][2], 100 * stack_item[idx][3]))
    #
    # print('Performance(COCOAPI): ')
    # for k, v in mAP.items():
    #     print('mAP@%s: %.1f' % (k, 100 * v))

    with open(result_file, 'r') as f:
        res = json.load(f)
        print("Summary len: {}".format(len(res)))