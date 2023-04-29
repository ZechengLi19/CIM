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
###################DEVKIT_DIR###########################################################

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from core.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'Annotation_file'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'

# Available datasets
DATASETS = {
    'coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco2014/train2017',
        ANN_FN:
            _DATA_DIR + '/coco2014/annotations/instances_train2017.json',
    },
    'coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco2014/val2017',
        ANN_FN:
            _DATA_DIR + '/coco2014/annotations/instances_val2017.json',
    },
    # 'coco_2017_test': {  # 2017 test uses 2015 test images
    #     IM_DIR:
    #         _DATA_DIR + '/coco2014/test2017',
    #     ANN_FN:
    #         _DATA_DIR + '/coco2014/annotations/image_info_test2017.json',
    #     IM_PREFIX:
    #         'COCO_test2015_'
    # },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco2014/test2017',
        ANN_FN:
            _DATA_DIR + '/coco2014/annotations/image_info_test-dev2017.json',
        # IM_PREFIX:
        #     'COCO_test2015_'
    },
    'voc_2012_sbdval': {
        # IM_DIR:
        #     _DATA_DIR + '/VOC2012/JPEGImages',
        # ANN_FN:
        #     _DATA_DIR + '/VOC2012/annotations/voc_2012_val.json',
        # DEVKIT_DIR:
        #     # _DATA_DIR + '/VOC2012/VOCdevkit'
        #     _DATA_DIR + '/VOC2012'
        IM_DIR:
            "/home/data2/Dataset/VOC2012/VOC2012_old/JPEGImages",

        ANN_FN:
            "/home/data2/Dataset/VOC2012/VOC2012_old/annotations/voc_2012_val.json",
        DEVKIT_DIR:
            "/home/data2/Dataset/VOC2012",
    },
    # 'voc_2012_sbdval_style': {
    #     IM_DIR:
    #         _DATA_DIR + '/VOC2012/JPEGImages',
    #     ANN_FN:
    #         _DATA_DIR + '/VOC2012/annotations/voc_2012_val_cocostyle.json',
    #     DEVKIT_DIR:
    #         _DATA_DIR + '/VOC2012/VOCdevkit'
    # },
    'voc_2012_trainaug': {
        IM_DIR:
            # _DATA_DIR + '/VOC2012/JPEGImages',
            "/home/data2/Dataset/VOC2012/VOC2012_old/JPEGImages",

        ANN_FN:
            # _DATA_DIR + '/VOC2012/annotations/voc_2012_trainaug.json',
            "/home/data2/Dataset/VOC2012/VOC2012_old/annotations/voc_2012_trainaug.json",
        DEVKIT_DIR:
            # _DATA_DIR + '/VOC2012/VOCdevkit'
            "/home/data2/Dataset/VOC2012/VOCdevkit",
    },
}
