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

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from six.moves import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse

# Must happen before importing COCO API (which imports matplotlib)
import utils.env as envu
envu.set_up_matplotlib()
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

import utils.boxes as box_utils
from core.config import cfg
from utils.timer import Timer

from .dataset_catalog import ANN_FN
from .dataset_catalog import DATASETS
from .dataset_catalog import IM_DIR
from .dataset_catalog import IM_PREFIX

'''
from lib.datasets.dataset_catalog import ANN_FN
from lib.datasets.dataset_catalog import DATASETS
from lib.datasets.dataset_catalog import IM_DIR
from lib.datasets.dataset_catalog import IM_PREFIX

category_ids = COCO.getCatIds()
categories = [c['name'] for c in COCO.loadCats(category_ids)]
category_to_id_map = dict(zip(categories, category_ids))
classes = categories
num_classes = len(classes)
json_category_id_to_contiguous_id = {v: i for i, v in enumerate(COCO.getCatIds())}
contiguous_category_id_to_json_id = { v: k for k, v in json_category_id_to_contiguous_id.items()}
classes = categories
num_classes = len(classes)


def _prep_roidb_entry(entry):
    """Adds empty metadata fields to an roidb entry."""
    # Reference back to the parent dataset
    entry['dataset'] = 'json_data'
    # Make file_name an abs path
    im_path = os.path.join(image_directory, image_prefix + entry['file_name'])
    assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
    entry['image'] = im_path
    entry['flipped'] = False
    # Empty placeholders
    entry['boxes'] = np.empty((0, 4), dtype=np.float32)  
    entry['masks'] = np.empty((0, 7, 7), dtype=np.float32)
    entry['mat'] = np.empty((0, 20), dtype=np.float32)  
    entry['gt_boxes'] = np.empty((0, 5), dtype=np.float32)
    entry['gt_classes'] = np.zeros((1, num_classes), dtype=np.int32)
    # Remove unwanted fields that come from the json file (if they exist)
    for k in ['date_captured', 'url', 'license', 'file_name']:
        if k in entry:
            del entry[k]

image_ids = COCO.getImgIds()   # 10582
image_ids.sort()
roidb = copy.deepcopy(COCO.loadImgs(image_ids))

for entry in roidb:
    _prep_roidb_entry(entry)


'''



logger = logging.getLogger(__name__)


class JsonDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name):
        assert name in DATASETS.keys(), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(DATASETS[name][IM_DIR]), \
            'Image directory \'{}\' not found'.format(DATASETS[name][IM_DIR])
        assert os.path.exists(DATASETS[name][ANN_FN]), \
            'Annotation file \'{}\' not found'.format(DATASETS[name][ANN_FN])
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.image_prefix = (
            '' if IM_PREFIX not in DATASETS[name] else DATASETS[name][IM_PREFIX]
        )
        self.COCO = COCO(DATASETS[name][ANN_FN])
        logger.info(DATASETS[name][ANN_FN])
        self.debug_timer = Timer()
        # Set up dataset classes
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i
            for i, v in enumerate(self.COCO.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache/lzc_path'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def get_roidb(
            self,
            gt=False,
            proposal_file=None,
            mat_file = None,
            min_proposal_size=2,
            proposal_limit=-1,
            crowd_filter_thresh=0
        ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        image_ids = self.COCO.getImgIds()   # 10582
        image_ids.sort()
        if cfg.DEBUG:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))[:100]
        else:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
            # im_id = COCO.getAnnIds( imgIds=image_ids[0])
            # box_gt = copy.deepcopy(COCO.loadAnns(im_id))
            
        for entry in roidb:
            self._prep_roidb_entry(entry)


        if gt:
            # disable cache lzc edit
            #
            # Include ground-truth object annotations
            cache_filepath = os.path.join(self.cache_path, self.name+'_gt_roidb_'+str(cfg.FAST_RCNN.MASK_SIZE)+'.pkl')
            if os.path.exists(cache_filepath) and not cfg.DEBUG :
                self.debug_timer.tic()
                logger.info('Loading cached gt_roidb from %s', cache_filepath)
                with open(cache_filepath, 'rb') as fp:
                    roidb = pickle.load(fp)
                logger.debug(
                    '_add_gt_from_cache took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
            else:
                self.debug_timer.tic()
                for entry in roidb:
                    self._add_gt_annotations(entry)
                logger.debug(
                    '_add_gt_annotations took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
                if not cfg.DEBUG:
                    print(cache_filepath)
                    with open(cache_filepath, 'wb') as fp:
                        pickle.dump(roidb, fp, pickle.HIGHEST_PROTOCOL)
                    logger.info('Cache ground truth roidb to %s', cache_filepath)

            ################
            #
            # self.debug_timer.tic()
            # for entry in roidb:
            #     self._add_gt_annotations(entry)
            # logger.debug(
            #     '_add_gt_annotations took {:.3f}s'.
            #     format(self.debug_timer.toc(average=False))
            # )

        if mat_file is not None:    
            self._add_prmmat_from_file(roidb, mat_file)   
        if proposal_file is not None:
            # Include proposals from a file
            self.debug_timer.tic()
            self._add_proposals_from_file(roidb, proposal_file, min_proposal_size, proposal_limit, crowd_filter_thresh)
            logger.debug(
                '_add_proposals_from_file took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
            
            
        return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        im_path = os.path.join(
            self.image_directory, self.image_prefix + entry['file_name']
        )
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['True_rois'] = np.empty((0, 4), dtype=np.float32)
        entry['masks'] = np.empty((0, cfg.FAST_RCNN.MASK_SIZE, cfg.FAST_RCNN.MASK_SIZE), dtype=np.float32) #��dtype=np.bool) #
        entry['gt_boxes'] = np.empty((0, 5), dtype=np.float32)
        entry['gt_classes'] = np.zeros((1, self.num_classes), dtype=np.int32)
        entry['mat'] = np.empty((0, self.num_classes + 1), dtype=np.float32)
        entry['iou_label'] = np.empty((0, 1), dtype=np.float32)
        entry['peak_score'] = np.empty((0, 1), dtype=np.float32)

        # Remove unwanted fields that come from the json file (if they exist)
        for k in ['date_captured', 'url', 'license', 'file_name']:
            if k in entry:
                del entry[k]

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            # crowd regions are RLE encoded and stored as dicts
            if obj['area'] < cfg.TRAIN.GT_MIN_AREA:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = box_utils.xywh_to_xyxy(obj['bbox'])
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                x1, y1, x2, y2, height, width
            )
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
            
            boxes = np.array([[x1, y1, x2, y2, self.json_category_id_to_contiguous_id[obj['category_id']]  ]])    
            entry['gt_boxes'] = np.append(entry['gt_boxes'],  boxes.astype(entry['gt_boxes'].dtype, copy=False),axis=0)    
                
        num_valid_objs = len(valid_objs)

        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            gt_classes[ix] = cls

        for cls in gt_classes:
            entry['gt_classes'][0, cls] = 1

    def _add_proposals_from_file(
        self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh
    ):
        """Add proposals from a proposals file to an roidb."""
        logger.info('Loading proposals from: {}'.format(proposal_file))
        with open(proposal_file, 'rb') as f:
            proposals = pickle.load(f)
        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
        if "True_rois" in proposals.keys():
            True_rois_flag = "True_rois"
            print("using True_rois")
        else:
            True_rois_flag = "boxes"
            print("using boxes replace True_rois")

        _sort_proposals(proposals, id_field)
        box_list = []
        True_rois_list = []
        mask_list = []
        for i, entry in enumerate(roidb):
            if i % 2500 == 0:
                logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
            boxes = proposals['boxes'][i]
            True_rois = proposals[True_rois_flag][i]

            #####################
            #coco
            #oldx1 = boxes[:, 1].copy()
            #oldx2 = boxes[:, 2].copy()
            #boxes[:, 1] = oldx2
            #boxes[:, 2] = oldx1
            
            ####################
            
            # Sanity check that these boxes are for the correct image id
            # assert entry['id'] == proposals[id_field][i]
            if not str(entry['id']) == str(proposals[id_field][i]):
                print(entry['id'])
                print(proposals[id_field][i])
                
                raise AssertionError
            # Remove duplicate boxes and very small boxes and then take top k
            boxes = box_utils.clip_boxes_to_image(boxes, entry['height'], entry['width'])
            True_rois = box_utils.clip_boxes_to_image(True_rois, entry['height'], entry['width'])

            # keep = box_utils.unique_boxes(boxes)
            # boxes = boxes[keep, :]
            # keep = box_utils.filter_small_boxes(boxes, min_proposal_size)
            # boxes = boxes[keep, :]
            if top_k > 0:
                boxes = boxes[:top_k, :]
                True_rois = True_rois[:top_k, :]
            box_list.append(boxes)
            True_rois_list.append(True_rois)
            
            masks = proposals['masks'][i]
            mask_list.append(masks)

        _merge_proposal_boxes_into_roidb(roidb, box_list, mask_list, True_rois_list)

    def _add_prmmat_from_file(
        self, roidb, mat_file
        ):
        """Add prm_scores from a prmmat file to an roidb."""
        logger.info('Loading proposals from: {}'.format(mat_file))
        with open(mat_file, 'rb') as f:
            prmmats = pickle.load(f)
        id_field = 'indexes' if 'indexes' in prmmats else 'ids'  # compat fix
        _sort_mats(prmmats, id_field)
        if 'peak_score' in prmmats.keys() and 'iou_label' in prmmats.keys():
            mat_list = []
            iou_label_list = []
            peak_score_list = []
            for i, entry in enumerate(roidb):
                if i % 2500 == 0:
                    logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
                mat = prmmats['mat'][i]
                iou_label = prmmats['iou_label'][i]
                peak_score = prmmats['peak_score'][i]
                # Sanity check that these boxes are for the correct image id
                # assert entry['id'] == proposals[id_field][i]
                if not str(entry['id']) == str(prmmats[id_field][i]):
                    print(entry['id'])
                    print(prmmats[id_field][i])
                    raise AssertionError
                mat_list.append(mat)
                iou_label_list.append(iou_label)
                peak_score_list.append(peak_score)
            _merge_mat_into_roidb(roidb, mat_list, iou_label_list,peak_score_list)
        elif 'iou_label' in prmmats.keys():
            mat_list = []
            iou_label_list = []
            for i, entry in enumerate(roidb):
                if i % 2500 == 0:
                    logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
                ###!!!
                mat = prmmats['mat'][i]  # .cpu().numpy()
                iou_label = prmmats['iou_label'][i]
                # Sanity check that these boxes are for the correct image id
                # assert entry['id'] == proposals[id_field][i]
                if not str(entry['id']) == str(prmmats[id_field][i]):
                    print(entry['id'])
                    print(prmmats[id_field][i])
                    raise AssertionError
                mat_list.append(mat)
                iou_label_list.append(iou_label)
            _merge_mat_into_roidb(roidb, mat_list, iou_label_list)
        else:
            mat_list = []
            for i, entry in enumerate(roidb):
                if i % 2500 == 0:
                    logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
                ###!!!
                mat = prmmats['mat'][i] # .cpu().numpy()
                # Sanity check that these boxes are for the correct image id
                # assert entry['id'] == proposals[id_field][i]
                if not str(entry['id']) == str(prmmats[id_field][i]):
                    print(entry['id'])
                    print(prmmats[id_field][i])
                    raise AssertionError
                mat_list.append(mat)
            _merge_mat_into_roidb(roidb, mat_list)

def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)


def _merge_proposal_boxes_into_roidb(roidb, box_list, mask_list, True_rois_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        True_rois = True_rois_list[i]
        masks = mask_list[i]

        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )

        entry['True_rois'] = np.append(
            entry['True_rois'],
            True_rois.astype(entry['True_rois'].dtype, copy=False),
            axis=0
        )

        # print(entry['masks'].shape, masks.shape)
        entry['masks'] = np.append(
            entry['masks'],
            masks.astype(entry['masks'].dtype, copy=False),
            axis=0
        )
        

def _merge_gt_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]

        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )


def _merge_mat_into_roidb(roidb, mat_list, iou_label_list = None, peak_score_list=None):
    """Add proposal boxes to each roidb entry."""
    if peak_score_list is None and iou_label_list is None:
        assert len(mat_list) == len(roidb)
        for i, entry in enumerate(roidb):
            mat = mat_list[i]
            entry['mat'] = np.append(entry['mat'],
                mat.astype(entry['mat'].dtype, copy=False),
                axis=0)

    elif not peak_score_list is None and not iou_label_list is None:
        assert len(mat_list) == len(iou_label_list) == len(peak_score_list) == len(roidb)
        for i, entry in enumerate(roidb):
            mat = mat_list[i]
            iou_label = iou_label_list[i]
            peak_score = np.array(peak_score_list[i])
            entry['mat'] = np.append(entry['mat'],
                                     mat.astype(entry['mat'].dtype, copy=False),
                                     axis=0)
            entry['iou_label'] = np.append(entry['iou_label'],
                                           iou_label.astype(entry['iou_label'].dtype, copy=False),
                                           axis=0)
            entry['peak_score'] = np.append(entry['peak_score'],
                                            peak_score.astype(entry['peak_score'].dtype, copy=False),
                                            axis=0)

    else:
        assert len(mat_list) == len(iou_label_list) == len(roidb)
        for i, entry in enumerate(roidb):
            mat = mat_list[i]
            iou_label = iou_label_list[i]
            entry['mat'] = np.append(entry['mat'],
                                     mat.astype(entry['mat'].dtype, copy=False),
                                     axis=0)
            entry['iou_label'] = np.append(entry['iou_label'],
                                           iou_label.astype(entry['iou_label'].dtype, copy=False),
                                           axis=0)




def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores','masks']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]

def _sort_mats(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['mat', id_field]
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]