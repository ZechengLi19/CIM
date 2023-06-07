import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname('__file__'))))
from lib.utils.mask_utils import mask_iou
import torch
from six.moves import cPickle as pickle
import cupy as cp
from scipy.io import loadmat
import scipy
from tqdm import tqdm

from pycocotools.coco import COCO
import multiprocessing
from pre_tools import *

def generate_coco2017(imgIds, cocoGt, data_dir):
    base_path = "./data/coco2017/COB-COCO"
    for index in tqdm(range(len(imgIds))):  # index = 0
        img_id = imgIds[index]

        # img
        path = cocoGt.loadImgs(img_id)[0]['file_name'] # .jpg

        # proposal
        file_name = 'COCO_train2014_' + path[:-4] + '.mat'
        if not os.path.exists(os.path.join(base_path, file_name)):
            file_name = 'COCO_val2014_' + path[:-4] + '.mat'
        if not os.path.exists(os.path.join(base_path, file_name)):
            file_name = path[:-4] + '.mat'

        if os.path.exists(os.path.join(data_dir, file_name.replace(".mat", "")[-12:] + ".pkl")):
            continue

        COB_proposals = scipy.io.loadmat(os.path.join(base_path, file_name),
            verify_compressed_data_integrity=False)['maskmat']

        mask_proposals = [np.array(p) for p in COB_proposals]
        mask_proposals = np.array(mask_proposals)
        num_proposal = len(mask_proposals)

        mask_proposals = cp.asarray(mask_proposals)

        iou_map = []
        for j in range(num_proposal):
            proposal_iou = mask_iou(mask_proposals, cp.expand_dims(mask_proposals[j], axis=0))
            iou_map.append(proposal_iou)

        iou_map = cp.asnumpy(cp.concatenate(iou_map,axis=1).astype(cp.float16))
        pickle.dump(iou_map, open(os.path.join(data_dir, file_name.replace(".mat", "")[-12:]+".pkl"), 'wb'), pickle.HIGHEST_PROTOCOL)

def generate_voc2012(imgIds, cocoGt, data_dir):
    for index in tqdm(range(len(imgIds))): # index = 0
        img_id = imgIds[index]

        s = str(int(img_id))
        file_name = s[:4] + '_' + s[4:]

        try:
            base_path = "./data/VOC2012/COB_SBD_trainaug"
            COB_proposals = loadmat( os.path.join(base_path,file_name+'.mat') )['maskmat'][:,0]
        except:
            base_path = "./data/VOC2012/COB_SBD_val"
            COB_proposals = loadmat( os.path.join(base_path,file_name+'.mat') )['maskmat'][:,0]

        mask_proposals = [np.array(p) for p in COB_proposals]
        mask_proposals = np.array(mask_proposals)
        num_proposal = len(mask_proposals)

        mask_proposals = cp.asarray(mask_proposals)

        iou_map = []
        for j in range(num_proposal):
            proposal_iou = mask_iou(mask_proposals, cp.expand_dims(mask_proposals[j], axis=0))
            iou_map.append(proposal_iou)

        iou_map = cp.asnumpy(cp.concatenate(iou_map,axis=1).astype(cp.float16))
        pickle.dump(iou_map, open(os.path.join(data_dir,file_name+".pkl"), 'wb'), pickle.HIGHEST_PROTOCOL)

               
if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset

    label_file_list = []

    if dataset == "voc":
        label_file_list.append("./data/VOC2012/annotations/voc_2012_trainaug.json")
        label_file_list.append("./data/VOC2012/annotations/voc_2012_val.json")
        data_dir = "./data/cob_iou/VOC2012"

    elif dataset == "coco":
        label_file_list.append("./data/coco2017/annotations/instances_train2017.json")
        label_file_list.append("./data/coco2017/annotations/instances_val2017.json")
        label_file_list.append("./data/coco2017/annotations/image_info_test-dev2017.json")
        data_dir = "./data/cob_iou/coco2017"

    else:
        raise NotImplementedError

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

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
            generate_fun = generate_voc2012
        else:
            generate_fun = generate_coco2017

        jobs = []
        for work_id in range(worker):
            if work_id + 1 != worker:
                p = multiprocessing.Process(target=generate_fun,
                                            args=(imgIds[work_id * per_len:(work_id + 1) * per_len],
                                                  cocoGt, data_dir))
            else:
                p = multiprocessing.Process(target=generate_fun,
                                            args=(imgIds[work_id * per_len:],
                                                  cocoGt,data_dir))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()

