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
import multiprocessing
import shutil
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine import empty_results, extend_results
from core.test import box_results_for_corloc, box_results_with_nms_and_limit
from datasets.json_dataset import JsonDataset
from datasets import task_evaluation
import utils.logging
from utils.mask_eval_utils import coco_encode, mask_results_with_nms_and_limit, \
    mask_results_with_nms_and_limit_get_index
from utils.mask_utils import mask_iou, mask_inside, mask_outside
from prm.coco_dataset import coco_nummap_id
from prm.prm_configs import ismember
from pycocotools import mask as maskUtils
import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils.boxes import expand_boxes, clip_boxes_to_image
import numpy as np
# from scipy.misc import imresize

import json
from pycocotools import mask as COCOMask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datasets import pycococreatortools
from datasets.json_inference import coco_inst_seg_eval

# from IPython import embed

cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate pseudo labels for Mask-RCNN training')
    parser.add_argument(
        '--dataset', required=True, choices=["voc2012trainaug", "coco2017train"],
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--result_path', required=True,
        default='./discovery.pkl',
        help='the path for result file.')
    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results.')
    parser.add_argument(
        '--is_best', default=False,
        help='output directory to save the testing results.')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')
    parser.add_argument(
        '--total_process', dest='total_process', default=24,type=float,
        help='number of processes')
    return parser.parse_args()

def eval(name, total,coco_output, roidb):
    instance_id = 1
    range_begin = int(name) * len(roidb) // total
    if (int(name) + 1) == total:
        range_end = len(roidb)
    else:
        range_end = (int(name) + 1) * len(roidb) // total
    print('Process%s: range from %d to %d' % (name, range_begin, range_end))
    for idx, entry in enumerate(roidb):
        if idx < int(name) * len(roidb) // total:
            continue
        if idx >= (int(name) + 1) * len(roidb) // total:
            continue
        boxes = all_boxes[entry['image']]
        scores = boxes['scores']

        (filepath, tempfilename) = os.path.split(entry['image'])

        if "coco" in args.dataset:
            base_path = "./data/coco2017/COB-COCO"

            file_n = entry['image'].split("/")[-1].replace(".jpg", ".mat")
            file_name = 'COCO_train2014_' + file_n
            if not os.path.exists(os.path.join(base_path, file_name)):
                file_name = 'COCO_val2014_' + file_n
            if not os.path.exists(os.path.join(base_path, file_name)):
                file_name = file_n
            mask_proposals = scipy.io.loadmat(
                os.path.join(base_path, file_name),
                verify_compressed_data_integrity=False)['maskmat'].copy()
            cls_num = 81

        else:
            file_name = entry['image'][-15:-4]
            try:
                base_path = "./data/VOC2012/COB_SBD_trainaug"
                mask_proposals = loadmat(os.path.join(base_path, file_name + '.mat'))['maskmat'][:, 0]
            except:
                base_path = "./data/VOC2012/COB_SBD_val"
                mask_proposals = loadmat(os.path.join(base_path, file_name + '.mat'))['maskmat'][:, 0]

            cls_num = 21

        img_id = int(entry['id'])
        DETECTIONS_PER_IM = 100
        scores, boxes, cls_boxes, cls_inds = mask_results_with_nms_and_limit_get_index(cfg, scores, boxes['boxes'],
                                                                                       DETECTIONS_PER_IM)

        img_size = (entry['width'], entry['height'])
        if cls_num == 21:
            image_info = pycococreatortools.create_image_info(img_id, entry['image'][-15:-4] + ".jpg", img_size)
        else:
            image_info = pycococreatortools.create_image_info(img_id, entry['image'][-16:-4] + ".jpg", img_size)

        coco_output["images"].append(image_info)

        for cls_idx in range(1, cls_num):
            if entry['gt_classes'][0][cls_idx - 1] > 0:
                cls_instance_idx = list(np.argsort(np.array(cls_boxes[cls_idx].copy()[:, 4])))
                cls_instance_idx.reverse()
                if len(cls_instance_idx) == 0:
                    continue
                best_score = cls_boxes[cls_idx][cls_instance_idx[0]][4].astype(np.float64)
                for i in range(len(cls_instance_idx)):
                    instance_idx = cls_instance_idx[i]
                    score = cls_boxes[cls_idx][instance_idx][4].astype(np.float64)
                    if not args.is_best:
                        if score == best_score:
                            if cls_num == 21:
                                category_id = int(cls_idx)
                            else:
                                category_id = coco_nummap_id[int(cls_idx) - 1]
                            COB_ind = cls_inds[cls_idx][instance_idx]
                            mask = mask_proposals[COB_ind]
                            category_info = {'id': category_id, 'is_crowd': False}

                            # best score keep
                            annotation_info = pycococreatortools.create_annotation_info_v1(
                                instance_id, img_id, category_info, mask,
                                np.asscalar(cls_boxes[cls_idx][instance_idx][4]),
                                (entry['width'], entry['height']), tolerance=0)
                            instance_id += 1
                            coco_output['annotations'].append(annotation_info)
                        else:
                            if cls_num == 21:
                                category_id = int(cls_idx)
                            else:
                                category_id = coco_nummap_id[int(cls_idx) - 1]
                            COB_ind = cls_inds[cls_idx][instance_idx]
                            mask = mask_proposals[COB_ind]
                            category_info = {'id': category_id, 'is_crowd': False}
                            annotation_info = pycococreatortools.create_annotation_info_v1(
                                instance_id, img_id, category_info, mask,
                                np.asscalar(cls_boxes[cls_idx][instance_idx][4]),
                                (entry['width'], entry['height']), tolerance=0)
                            instance_id += 1
                            coco_output['annotations'].append(annotation_info)
                    else:
                        if score == best_score:
                            if cls_num == 21:
                                category_id = int(cls_idx)
                            else:
                                category_id = coco_nummap_id[int(cls_idx) - 1]
                            COB_ind = cls_inds[cls_idx][instance_idx]
                            mask = mask_proposals[COB_ind]
                            category_info = {'id': category_id, 'is_crowd': False}
                            annotation_info = pycococreatortools.create_annotation_info_v1(
                                instance_id, img_id, category_info, mask,
                                np.asscalar(cls_boxes[cls_idx][instance_idx][4]),
                                (entry['width'], entry['height']), tolerance=0)
                            instance_id += 1
                            coco_output['annotations'].append(annotation_info)
        print(f'\rImage Index: {(idx + 1):.0f}/{num_images:.0f}  ', end='')
    save_dir = os.path.join(args.output_dir, 'tmp')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not args.is_best:
        with open(os.path.join(save_dir, 'msrcnn_pseudo_label' + '_' + str(range_begin) + '_' + str(range_end) +'.json'), 'w') as f:
            f.write(json.dumps(coco_output))
    else:
        with open(os.path.join(save_dir, 'msrcnn_pseudo_label_best'+ '_' + str(range_begin) + '_' + str(range_end) +'.json'), 'w') as f:
            f.write(json.dumps(coco_output))


if __name__ == '__main__':
    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    if args.result_path:
        while not os.path.exists(args.result_path):
            logger.info('Waiting for {} to exist...'.format(args.result_path))
            time.sleep(10)
    assert os.path.exists(args.result_path)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.result_path)
        logger.info('Automatically set output directory to %s', args.output_dir)

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "coco2017train":
        cfg.TEST.DATASETS = ('coco_2017_train',)
        cfg.MODEL.NUM_CLASSES = 80
        cfg.TEST.PROPOSAL_FILES = cfg.TRAIN.PROPOSAL_FILES
        annFile="./data/coco2017/annotations/instances_train2017.json"
        val_json = json.load(open('./data/coco2017/annotations/instances_val2017.json'))

    elif args.dataset == 'voc2012trainaug':
        cfg.TEST.DATASETS = ('voc_2012_trainaug',)
        cfg.MODEL.NUM_CLASSES = 20
        cfg.TEST.PROPOSAL_FILES = cfg.TRAIN.PROPOSAL_FILES
        annFile = "./data/VOC2012/annotations/voc_2012_trainaug.json"
        val_json = json.load(open('./data/VOC2012/annotations/voc_2012_val.json'))

    else:
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()

    logger.info('Re-evaluation with config:')
    logger.info(pprint.pformat(cfg))

    with open(args.result_path, 'rb') as f:
        results = pickle.load(f)
        logger.info('Loading results from {}.'.format(args.result_path))
    all_boxes = results['all_boxes']

    dataset_name = cfg.TEST.DATASETS[0]
    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb(gt=True)
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES + 1

    coco_output = {}
    coco_output["images"] = []
    coco_output["annotations"] = []
    coco_output['categories'] = val_json['categories']

    total_process = args.total_process
    jobs = []
    for i in range(total_process):
        p = multiprocessing.Process(target=eval, args=(str(i), total_process, coco_output, roidb))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()

    list_all = os.listdir(os.path.join(args.output_dir,'tmp'))
    block = {}
    for name in list_all:
        with open(os.path.join(args.output_dir, 'tmp', name), 'rb') as f:
            block[name] = json.load(f)

            os.remove(os.path.join(args.output_dir, 'tmp', name))

    max_instance_id = 0
    for i in range(total_process):
        range_begin = int(i) * len(roidb) // total_process
        if (int(i) + 1) == total_process:
            range_end = len(roidb)
        else:
            range_end = (int(i) + 1) * len(roidb) // total_process
        if not args.is_best:
            key = 'msrcnn_pseudo_label'+ '_' + str(range_begin) + '_' + str(range_end) + ".json"
        else:
            key = 'msrcnn_pseudo_label_best' + '_' + str(range_begin) + '_' + str(range_end) + ".json"
        coco_output['images'].extend(block[key]['images'])

        coco_output['annotations'].extend(block[key]['annotations'])
        for i in block[key]['annotations']:
            i['id'] = i['id'] + max_instance_id
        max_instance_id = len(coco_output['annotations'])

    shutil.rmtree(os.path.join(args.output_dir, 'tmp'))

    if not args.is_best:
        with open(os.path.join(args.output_dir, 'msrcnn_pseudo_label.json'), 'w') as f:
            f.write(json.dumps(coco_output))

    else:
        with open(os.path.join(args.output_dir, 'msrcnn_pseudo_label_best.json'), 'w') as f:
            f.write(json.dumps(coco_output))

    # with open(os.path.join(args.output_dir, 'temp.json'), 'w') as f:
    #     f.write(json.dumps(coco_output['annotations']))
    #
    # result_file = os.path.join(args.output_dir, 'temp.json')
    #
    # annType = ['segm', 'bbox', 'keypoints']
    # annType = annType[0]
    # cocoGt = COCO(annFile)
    # cocoDt = cocoGt.loadRes(result_file)
    # cocoEval = COCOeval(cocoGt, cocoDt, annType)
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()
    #
    # mAP, cls_ap, cls_names = coco_inst_seg_eval(annFile, result_file)
    #
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
    #
    # with open(result_file, 'r') as f:
    #     res = json.load(f)
    #     print("Summary len: {}".format(len(res)))
