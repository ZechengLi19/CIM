import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname('__file__'))))
from lib.prm.prm_model_gt import peak_response_mapping, fc_resnet50
from lib.prm.prm_configs import open_transform
from lib.utils.mask_utils import mask_iou

import torch
from six.moves import cPickle as pickle
from scipy.io import loadmat
import scipy
from tqdm import tqdm
from pycocotools.coco import COCO
import multiprocessing
import copy

from pre_tools import *

trash="./data/trash"
useless_file = "assignment_label_rm_{}.pkl"
os.makedirs(trash, exist_ok=True)

def assign_voc2012(imgIds, worker_id, dataset, cocoGt):
    img_dir = "./data/VOC2012/JPEGImages"
    cob_original_file = "./data/VOC2012/COB_SBD_trainaug"
    point_level_file = './data/VOC2012/Center_points'
    sbd_proposals = dict(indexes=[], mat=[], score=[])
    for index in tqdm(range(len(imgIds))):
        img_id = imgIds[index]
        cluster_idx = 1

        # img
        s = str(int(img_id))
        file_name = s[:4] + '_' + s[4:]

        # proposal
        COB_proposals = loadmat(os.path.join(cob_original_file, file_name + '.mat'))['maskmat'][:, 0]

        mask_proposals = [np.array(p) for p in COB_proposals]
        mask_proposals = np.array(mask_proposals)
        num_proposal = len(mask_proposals)

        # label
        ann_ids = cocoGt.getAnnIds(imgIds=img_id)
        anns = cocoGt.loadAnns(ann_ids)
        boxes_cl = [ann['category_id'] - 1 for ann in anns]
        boxes_cl = torch.tensor(boxes_cl).unique()

        txt_file = os.path.join(point_level_file, file_name + '.txt')

        # model
        with open(txt_file, 'r') as pf:
            points = pf.read().splitlines()
            points = [p.strip().split(" ") for p in points]
            points = [[float(p[0]), float(p[1]), int(p[2]), float(p[3])] for p in points]
            # point (x_coord, y_coord, class-idx, conf)
        label_assignment = np.zeros((num_proposal, 21), dtype=np.float32)

        if len(points) == 0:
            label_assignment[label_assignment.sum(1) == 0, 0] = cluster_idx

            sbd_proposals['indexes'].append(img_id)
            sbd_proposals['mat'].append(label_assignment)
        else:
            bg_ind_agg = np.zeros(num_proposal,dtype=np.float32)
            for j in range(len(points)):  # j= 0
                class_idx = points[j][2]

                assert class_idx in boxes_cl

                x = int(points[j][0])
                y = int(points[j][1])

                peak_pass_proposal_idx = mask_proposals[:, y, x] > 0

                # submask assign
                avgmask = mask_proposals[peak_pass_proposal_idx, :, :].mean(0) > 0.7
                proposal_iou = mask_iou(mask_proposals, np.expand_dims(avgmask, axis=0))
                assign_ind = proposal_iou[:, 0] > 0.5

                label_assignment[assign_ind, :] = 0
                label_assignment[assign_ind, class_idx + 1] = cluster_idx

                bg_ind = (proposal_iou[:,0] <= 0.5).astype(np.float32) * \
                         (proposal_iou[:,0] != 0).astype(np.float32)

                bg_ind_agg += bg_ind

                cluster_idx += 1

            bg_ind_agg = (bg_ind_agg != 0).astype(np.float32) * (label_assignment.sum(1) == 0).astype(np.float32)
            label_assignment[bg_ind_agg != 0, 0] = cluster_idx

            print((label_assignment.sum(1) == 0).sum())

            sbd_proposals['indexes'].append(img_id)
            sbd_proposals['mat'].append(label_assignment)


    pickle.dump(sbd_proposals, open(
        os.path.join(trash, useless_file.format(worker_id)), 'wb'),
                pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = parse_args()

    dataset = args.dataset

    if dataset == "voc":
        label_file = "./data/VOC2012/annotations/voc_2012_trainaug.json"
        assignment_label_file = "./data/label_assign/voc_2012_point_label_assign.pkl"
    elif dataset == "coco":
        raise NotImplementedError
    else:
        raise NotImplementedError

    print(label_file)
    print(assignment_label_file)
    ###
    worker = 32

    cocoGt = COCO(label_file)
    imgIds = sorted(cocoGt.getImgIds())

    per_len = int(len(imgIds) / worker)

    if dataset == "voc":
        assign_fun = assign_voc2012
    else:
        raise NotImplementedError

    jobs = []
    for worker_id in range(worker):
        if worker_id + 1 != worker:
            p = multiprocessing.Process(target=assign_fun,
                                        args=(imgIds[worker_id * per_len:(worker_id + 1) * per_len], worker_id,
                                              dataset, cocoGt))
        else:
            p = multiprocessing.Process(target=assign_fun,
                                        args=(imgIds[worker_id * per_len:], worker_id,
                                              dataset, cocoGt))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()

    res = dict(indexes=[], mat=[], score=[])
    worker_id = 0
    while (worker_id != worker):
        path = os.path.join(trash, useless_file.format(worker_id))
        try:
            result = pickle.load(open(path, 'rb'))
            res['indexes'] += result['indexes']
            res['mat'] += result['mat']
            os.remove(path)

            worker_id += 1
        except:
            pass

    print("imgs len: " + str(len(imgIds)))
    print("indexes len: " + str(len(res["indexes"])))
    print("mat len: " + str(len(res["mat"])))

    pickle.dump(res, open(assignment_label_file, 'wb'), pickle.HIGHEST_PROTOCOL)


