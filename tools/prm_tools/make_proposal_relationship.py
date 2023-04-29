# -*- coding: utf-8 -*-

import os
import sys
# sys.path.append("lib/datasets/")
# sys.path.append("lib/prm/")
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname('__file__'))))
from lib.datasets.voc_data import VOC_CLASSES_background,VOC_COLORS_background
from lib.prm.prm_model_gt import peak_response_mapping, fc_resnet50
from lib.prm.prm_configs import open_transform
from lib.utils.mask_utils import mask_iou,mask_asymmetric_iou
from lib.prm.prm_configs import ismember

import torch
# from torchvision import ops
import torch.nn.functional as F

import json
from six.moves import cPickle as pickle

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from scipy.io import loadmat
# from scipy.misc import imresize
import scipy
import yaml
from tqdm import tqdm

from pycocotools.coco import COCO
import multiprocessing
import copy
print(multiprocessing.get_start_method())
multiprocessing.set_start_method('spawn', force=True)

# edit
# data_dir = "/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/data/cob_iou/VOC2012"
# data_dir = "/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/data/cob_asy_iou/VOC2012"
data_dir = "/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/data/cob_iou/coco2017"
# data_dir = "/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/data/cob_asy_iou/coco2017"
dataset = "coco"
print(data_dir)
ASY = 'asy' in data_dir
print(ASY)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

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

def imresize(arr,size,interp='bilibear',mode=None):
    im = Image.fromarray(np.uint8(arr),mode=mode)
    ts = type(size)
    if np.issubdtype(ts,np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size)*percent).astype(int))
    elif np.issubdtype(type(size),np.floating):
        size = tuple((np.array(im.size)*size).astype(int))
    else:
        size = (size[1],size[0])
    func = {'nearest':0,'lanczos':1,'biliear':2,'bicubic':3,'cubic':3}
    imnew = im.resize(size,resample=func[interp])   
    return np.array(imnew)

def generate_assugn_coco2017(imgIds,yml_par,dataset,cocoGt):
    # yml_par[dataset]["val_proposals"] == yml_par[dataset]["train_proposals"]
    cob_original_file = yml_par[dataset]["val_proposals"]
    for index in tqdm(range(len(imgIds))):  # index = 0
        img_id = imgIds[index]

        # img
        path = cocoGt.loadImgs(img_id)[0]['file_name'] # .jpg

        # proposal
        file_name = 'COCO_train2014_' + path[:-4] + '.mat'
        if not os.path.exists(os.path.join(cob_original_file, file_name)):
            file_name = 'COCO_val2014_' + path[:-4] + '.mat'
        if not os.path.exists(os.path.join(cob_original_file, file_name)):
            file_name = path[:-4] + '.mat'

        if os.path.exists(os.path.join(data_dir, file_name.replace(".mat", "")[-12:] + ".pkl")):
            continue

        COB_proposals = scipy.io.loadmat(os.path.join(cob_original_file, file_name),
            verify_compressed_data_integrity=False)['maskmat']

        mask_proposals = [np.array(p) for p in COB_proposals]
        mask_proposals = np.array(mask_proposals)
        num_proposal = len(mask_proposals)

        mask_proposals = cp.asarray(mask_proposals)

        iou_map = []
        for j in range(num_proposal): #  j= 0
            avgmask = mask_proposals[j]

            # cupy
            if ASY == True:
                proposal_iou = mask_asymmetric_iou(mask_proposals,cp.expand_dims(avgmask,axis=0))
            else:
                proposal_iou = mask_iou(mask_proposals, cp.expand_dims(avgmask, axis=0))

            iou_map.append(proposal_iou)

        # cupy
        iou_map = cp.asnumpy(cp.concatenate(iou_map,axis=1).astype(cp.float16))

        pickle.dump(iou_map, open(os.path.join(data_dir, file_name.replace(".mat", "")[-12:]+".pkl"), 'wb'), pickle.HIGHEST_PROTOCOL)

def generate_assign_voc2012(imgIds,yml_par,dataset,cocoGt):
    for index in tqdm(range(len(imgIds))): # index = 0
        img_id = imgIds[index]

        # img
        s = str(int(img_id))
        file_name = s[:4] + '_' + s[4:]

        # proposal
        try:
            COB_proposals = loadmat( os.path.join(yml_par[dataset]["train_proposals"],file_name+'.mat') )['maskmat'][:,0]
        except:
            COB_proposals = loadmat( os.path.join(yml_par[dataset]["val_proposals"],file_name+'.mat') )['maskmat'][:,0]

        mask_proposals = [np.array(p) for p in COB_proposals]
        mask_proposals = np.array(mask_proposals)
        num_proposal = len(mask_proposals)

        mask_proposals = cp.asarray(mask_proposals)

        iou_map = []
        for j in range(num_proposal): #  j= 0
            avgmask = mask_proposals[j]

            # cupy
            if ASY == True:
                proposal_iou = mask_asymmetric_iou(mask_proposals,cp.expand_dims(avgmask,axis=0))
            else:
                proposal_iou = mask_iou(mask_proposals, cp.expand_dims(avgmask, axis=0))

            iou_map.append(proposal_iou)

        iou_map = cp.asnumpy(cp.concatenate(iou_map,axis=1).astype(cp.float16))

        pickle.dump(iou_map, open(os.path.join(data_dir,file_name+".pkl"), 'wb'), pickle.HIGHEST_PROTOCOL)

               
if __name__ == '__main__':
    label_file_list = []

    with open("/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/configs/file_path/file_paths.yaml") as f:
        yml_par = yaml.load(f)

    label_file_list.append(yml_par[dataset]["train_json_file"])
    label_file_list.append(yml_par[dataset]["val_json_file"])

    try:
        label_file_list.append(yml_par[dataset]["test_json_file"])

    except:
        pass

    print(label_file_list)

    ###
    worker = 24

    for label_file in label_file_list:
        print(label_file)
        cocoGt = COCO(label_file)
        imgIds = sorted(cocoGt.getImgIds())

        per_len = int(len(imgIds) / worker)

        n_gpus = torch.cuda.device_count()

        if dataset == "voc":
            generate_assign_function = generate_assign_voc2012
        else:
            generate_assign_function = generate_assugn_coco2017

        jobs = []
        for lzc_idx in range(worker):
            if lzc_idx + 1 != worker:
                p = multiprocessing.Process(target=generate_assign_function,
                                            args=(imgIds[lzc_idx * per_len:(lzc_idx + 1) * per_len],
                                                  yml_par,dataset,cocoGt))
            else:
                p = multiprocessing.Process(target=generate_assign_function,
                                            args=(imgIds[lzc_idx * per_len:],
                                                  yml_par,dataset,cocoGt))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()

