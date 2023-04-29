# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

def invert_dict(d):
    return dict([(v,k) for (k,v) in d.items()])

def coco_object_categories():
    coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
    coco_id_num_map={1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
                   6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
                   11:10, 13: 11, 14: 12, 15: 13,
                   16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
                   22: 20, 23: 21, 24: 22, 25: 23, 27: 24,
                   28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
                   35: 30, 36:31, 37: 32, 38: 33, 39: 34,
                   40: 35, 41: 36, 42: 37, 43: 38,
                   44: 39, 46: 40, 47: 41, 48: 42, 49:43, 50: 44,
                   51: 45, 52:46, 53: 47, 54: 48, 55: 49,
                   56: 50, 57:51, 58: 52, 59: 53, 60: 54,
                   61:55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60,
                   70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66,
                   77: 67, 78: 68, 79: 69, 80: 70, 81: 71,
                   82: 72, 84: 73, 85: 74, 86: 75, 87: 76,
                   88: 77, 89: 78, 90: 79}
    return coco_id_num_map


coco_nummap_id = invert_dict(coco_object_categories())
coco_id_num_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
                   6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
                   11: 10, 13: 11, 14: 12, 15: 13,
                   16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
                   22: 20, 23: 21, 24: 22, 25: 23, 27: 24,
                   28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
                   35: 30, 36: 31, 37: 32, 38: 33, 39: 34,
                   40: 35, 41: 36, 42: 37, 43: 38,
                   44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44,
                   51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
                   56: 50, 57: 51, 58: 52, 59: 53, 60: 54,
                   61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60,
                   70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66,
                   77: 67, 78: 68, 79: 69, 80: 70, 81: 71,
                   82: 72, 84: 73, 85: 74, 86: 75, 87: 76,
                   88: 77, 89: 78, 90: 79}

#############################COCO#################################
class COCO_Classification(Dataset):   ## Dataset for COCO classification.        read the dataset, complete the annotation initialization
    
    def __init__(self, data_dir="/mass/wsk/dataset/coco2014/train2014",
                 annFile = "/mass/wsk/dataset/coco2014/annotations/instances_train2014.json" ,
                 classes = coco_object_categories(),
                 transform=None, target_transform=None):
        self.data_dir = data_dir
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.classes = classes
        self.transform = transform
        self.target_transform = target_transform
  
    def __getitem__(self, index):
        classes = self.classes
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # fet the target
        target_det = coco.loadAnns(ann_ids)
        # target = np.zeros(80)
        target = np.full((80),0)
        for obj_idx,obj in enumerate(target_det):  #range(len(target_det)):
            target[classes[obj['category_id']]] = 1
        target = torch.from_numpy(target).float()
        # open the img
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.data_dir, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.data_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str    
    