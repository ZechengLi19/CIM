# -*- coding: utf-8 -*-

import os
import numpy as np
from six.moves import cPickle as pickle
from scipy.io import loadmat
# from scipy.misc import imresize
import matplotlib.pyplot as plt
import cv2
import sys

sys.path.append("lib/utils/")
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname('__file__'))))
from lib.utils.box_utils import find_bbox_with_outlier
# from lib.prm.prm_configs import ismember
import scipy
import yaml
from PIL import Image

from pycocotools.coco import COCO
import multiprocessing
from tqdm import tqdm

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
    imnew = im.resize(size,resample=func[interp])    # 调用PIL库中的resize函数
    return np.array(imnew)

def generate_pkl_voc2012(imgIds,lzc_idx):
    SBD_mask_val_proposals = dict(indexes=[], masks=[], boxes=[], scores=[])
    # for index in range(len(imgIds)):  # index = 0
    for index in tqdm(range(len(imgIds))):  # index = 0
        img_id = imgIds[index]

        s = str(int(img_id))
        file_name = s[:4] + '_' + s[4:]

        # COB_proposals = loadmat( os.path.join( cob_original_file, file_name+'.mat' ) )['maskmat'][:,0]
        if os.path.exists(os.path.join(train_proposals_dir, file_name + '.mat')) == False:
            cob_original_file = val_proposals_dir
        else:
            cob_original_file = train_proposals_dir

        COB_proposals = scipy.io.loadmat(os.path.join(cob_original_file, file_name + '.mat'))['maskmat'][:, 0]

        boxes = np.empty((0, 4), dtype=np.uint16)
        masks = np.empty((0, mask_size, mask_size), dtype=np.bool)
        scores = np.zeros(len(COB_proposals))

        for pro_ind in range(len(COB_proposals)):
            ind_xy = np.nonzero(COB_proposals[pro_ind])
            xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
            mask = COB_proposals[pro_ind][ymin:ymax, xmin:xmax]
            mask = imresize(mask.astype(int), (mask_size, mask_size), interp='nearest')

            boxes = np.append(boxes, np.array([[xmin, ymin, xmax, ymax]], dtype=np.uint16), axis=0)
            masks = np.append(masks, mask[np.newaxis, :].astype(bool), axis=0)
        SBD_mask_val_proposals['indexes'].append(img_id)
        SBD_mask_val_proposals['masks'].append(masks)
        SBD_mask_val_proposals['boxes'].append(boxes)
        SBD_mask_val_proposals['scores'].append(scores)
        print(f'\rImage Index: {(index + 1):.0f}/{len(imgIds):.0f}  ', end='')
    pickle.dump(SBD_mask_val_proposals, open(os.path.join("/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/data/trash","voc_{}.pkl".format(lzc_idx)), 'wb'), pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    mask_size = 7
    max_proposal_num = 200

    with open("/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/configs/file_path/file_paths.yaml") as f:
        yml_par = yaml.load(f)

    train_proposals_dir = yml_par["voc"]["train_proposals"]
    val_proposals_dir = yml_par["voc"]["val_proposals"]
    # data
    for i in range(2):
        if i == 0:
            # train
            label_file = yml_par["voc"]["train_json_file"]
            cob_propsal_file = yml_par["voc"]["train_cob_proposal_file"]

        else:
            # val
            label_file = yml_par["voc"]["val_json_file"]
            cob_propsal_file = yml_par["voc"]["val_cob_proposal_file"]

        print(label_file)
        print(cob_propsal_file)

        worker = 48

        cocoGt = COCO(label_file)
        imgIds = sorted(cocoGt.getImgIds())

        per_len = int(len(imgIds)/worker)

        jobs = []
        for lzc_idx in range(worker):
            if lzc_idx + 1 != worker:
                p = multiprocessing.Process(target=generate_pkl_voc2012, args=(imgIds[lzc_idx*per_len:(lzc_idx+1)*per_len],lzc_idx))

            else:
                p = multiprocessing.Process(target=generate_pkl_voc2012, args=(imgIds[lzc_idx * per_len:], lzc_idx))

            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()

        res = dict(indexes=[], masks=[], boxes=[], scores=[])
        lzc_idx = 0
        while(lzc_idx != worker):
            path = os.path.join("/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/data/trash", "voc_{}.pkl".format(lzc_idx))
            try:
                result = pickle.load(open(path, 'rb'))

                for key in res.keys():
                    res[key] += result[key]

                os.remove(path)

                lzc_idx += 1
            except:
                pass

        print("imgs len: "+str(len(imgIds)))

        for key in res.keys():
            print("{} len: ".format(key) + str(len(res[key])))

        pickle.dump(res, open(cob_propsal_file,'wb'), pickle.HIGHEST_PROTOCOL)
