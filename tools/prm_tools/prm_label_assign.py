import os
import sys

# sys.path.append("lib/datasets/")
# sys.path.append("lib/prm/")
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname('__file__'))))
from lib.datasets.voc_data import VOC_CLASSES_background, VOC_COLORS_background
from lib.prm.prm_model_gt import peak_response_mapping, fc_resnet50  # 非原来的 PRM
from lib.prm.prm_configs import open_transform
from lib.utils.mask_utils import mask_iou
from lib.prm.prm_configs import ismember

import torch
# from torchvision import ops
import torch.nn.functional as F

import json
from six.moves import cPickle as pickle

import numpy as np
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

useless_file = "assignment_label_rm_{}.pkl"

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


def imresize(arr, size, interp='bilibear', mode=None):
    im = Image.fromarray(np.uint8(arr), mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size) * percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'biliear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])  # 调用PIL库中的resize函数
    return np.array(imnew)


def generate_assugn_coco2017(imgIds, lzc_idx, model, device, yml_par, dataset, cocoGt):
    img_dir = yml_par[dataset]["img_dir_train"]
    cob_original_file = yml_par[dataset]["train_proposals"]
    model = model.inference().to(device)
    sbd_proposals = dict(indexes=[], mat=[], score=[])
    for index in tqdm(range(len(imgIds))):  # index = 0
        img_id = imgIds[index]
        cluster_idx = 1

        # img
        path = cocoGt.loadImgs(img_id)[0]['file_name']  # .jpg

        img = Image.open(os.path.join(img_dir, path)).convert('RGB')
        inputs = open_transform(img).unsqueeze(0).to(device)  # 原来的大小

        # proposal
        file_name = 'COCO_train2014_' + path[:-4] + '.mat'
        if not os.path.exists(os.path.join(cob_original_file, file_name)):
            file_name = 'COCO_val2014_' + path[:-4] + '.mat'
        if not os.path.exists(os.path.join(cob_original_file, file_name)):
            file_name = path[:-4] + '.mat'
        COB_proposals = scipy.io.loadmat(os.path.join(cob_original_file, file_name),
                                         verify_compressed_data_integrity=False)['maskmat']

        mask_proposals = [np.array(p) for p in COB_proposals]
        mask_proposals = np.array(mask_proposals)
        num_proposal = len(mask_proposals)

        # label
        ann_ids = cocoGt.getAnnIds(imgIds=img_id)
        anns = cocoGt.loadAnns(ann_ids)
        boxes_cl = [coco_id_num_map[ann['category_id']] for ann in anns]
        boxes_cl = torch.tensor(boxes_cl).unique().to(device)

        # model
        visual_cues = model(inputs, boxes_cl, class_threshold=0, peak_threshold=10)
        label_assignment = np.zeros((num_proposal, 81), dtype=np.float32)  # 0-80: 0 for bg, 1-80:fg

        if visual_cues == None:
            label_assignment[label_assignment.sum(1) == 0, 0] = cluster_idx

            sbd_proposals['indexes'].append(img_id)
            sbd_proposals['mat'].append(label_assignment)
        else:
            aggregation, class_response_maps, valid_peak_list, peak_response_maps, peak_score = visual_cues

            order = peak_score.numpy().argsort()
            valid_peak_list = valid_peak_list[order]
            peak_score = peak_score[order]

            peak_list = valid_peak_list.cpu().numpy()  # (1, 4)

            bg_ind_agg = np.zeros(num_proposal,dtype=np.float32)
            for j in range(len(peak_response_maps)):  # j= 0

                class_idx = peak_list[j, 1]
                # mask_proposals: 200,375,500
                x = int(peak_list[j, 2] * mask_proposals.shape[1] / 112)
                y = int(peak_list[j, 3] * mask_proposals.shape[2] / 112)
                avgmask = mask_proposals[mask_proposals[:, x, y] > 0, :, :].mean(0) > 0.7

                proposal_iou = mask_iou(mask_proposals, np.expand_dims(avgmask, axis=0))

                assign_ind = proposal_iou[:, 0] > 0.5
                label_assignment[assign_ind, :] = 0
                label_assignment[assign_ind, class_idx + 1] = cluster_idx

                bg_ind = (proposal_iou[:, 0] <= 0.5).astype(np.float32) * \
                         (proposal_iou[:, 0] != 0).astype(np.float32)

                bg_ind_agg += bg_ind

                cluster_idx += 1

            # label_assignment[label_assignment.sum(1) == 0, 0] = cluster_idx

            bg_ind_agg = (bg_ind_agg != 0).astype(np.float32) * (label_assignment.sum(1) == 0).astype(np.float32)
            label_assignment[bg_ind_agg != 0, 0] = cluster_idx

            print((label_assignment.sum(1) == 0).sum())

            sbd_proposals['indexes'].append(img_id)
            sbd_proposals['mat'].append(label_assignment)
        print(f'\rImage Index: {(index + 1):.0f}/{len(imgIds):.0f}  ', end='')

    pickle.dump(sbd_proposals, open(
        os.path.join("/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/data/trash", useless_file.format(lzc_idx)), 'wb'),
                pickle.HIGHEST_PROTOCOL)


def generate_assign_voc2012(imgIds, lzc_idx, model, device, yml_par, dataset, cocoGt):
    img_dir = yml_par[dataset]["img_dir_train"]
    cob_original_file = yml_par[dataset]["train_proposals"]
    model = model.inference().to(device)
    # sbd_proposals = dict(indexes=[], mat=[], score=[], iou_label=[])
    sbd_proposals = dict(indexes=[], mat=[], score=[])
    for index in tqdm(range(len(imgIds))):  # index = 0
        img_id = imgIds[index]
        cluster_idx = 1

        # img
        s = str(int(img_id))
        file_name = s[:4] + '_' + s[4:]
        img = Image.open(os.path.join(img_dir, file_name + '.jpg')).convert('RGB')
        inputs = open_transform(img).unsqueeze(0).to(device)  # 原来的大小

        # proposal
        COB_proposals = loadmat(os.path.join(cob_original_file, file_name + '.mat'))['maskmat'][:, 0]

        mask_proposals = [np.array(p) for p in COB_proposals]
        mask_proposals = np.array(mask_proposals)
        num_proposal = len(mask_proposals)

        # label
        ann_ids = cocoGt.getAnnIds(imgIds=img_id)
        anns = cocoGt.loadAnns(ann_ids)
        boxes_cl = [ann['category_id'] - 1 for ann in anns]
        boxes_cl = torch.tensor(boxes_cl).unique().to(device)

        # model
        visual_cues = model(inputs, boxes_cl, class_threshold=0, peak_threshold=10)
        label_assignment = np.zeros((num_proposal, 21), dtype=np.float32)  # 0-80: 0 for bg, 1-80:fg
        # iou_label = np.zeros((num_proposal, 1), dtype=np.float32)  # 0 - 1

        if visual_cues == None:
            label_assignment[label_assignment.sum(1) == 0, 0] = cluster_idx

            sbd_proposals['indexes'].append(img_id)
            sbd_proposals['mat'].append(label_assignment)
            # sbd_proposals['iou_label'].append(iou_label)
        else:
            aggregation, class_response_maps, valid_peak_list, peak_response_maps, peak_score = visual_cues

            order = peak_score.numpy().argsort()
            valid_peak_list = valid_peak_list[order]
            peak_score = peak_score[order]

            peak_list = valid_peak_list.cpu().numpy()  # (1, 4)

            bg_ind_agg = np.zeros(num_proposal,dtype=np.float32)
            for j in range(len(peak_response_maps)):  # j= 0
                class_idx = peak_list[j, 1]

                x = int(peak_list[j, 2] * mask_proposals.shape[1] / 112)
                y = int(peak_list[j, 3] * mask_proposals.shape[2] / 112)

                peak_pass_proposal_idx = mask_proposals[:, x, y] > 0

                # submask assign
                avgmask = mask_proposals[peak_pass_proposal_idx, :, :].mean(0) > 0.7
                proposal_iou = mask_iou(mask_proposals, np.expand_dims(avgmask, axis=0))
                assign_ind = proposal_iou[:, 0] > 0.5

                label_assignment[assign_ind, :] = 0
                label_assignment[assign_ind, class_idx + 1] = cluster_idx
                # iou_label[assign_ind, 0] = 1

                bg_ind = (proposal_iou[:,0] <= 0.5).astype(np.float32) * \
                         (proposal_iou[:,0] != 0).astype(np.float32)

                bg_ind_agg += bg_ind

                cluster_idx += 1

            bg_ind_agg = (bg_ind_agg != 0).astype(np.float32) * (label_assignment.sum(1) == 0).astype(np.float32)
            label_assignment[bg_ind_agg != 0, 0] = cluster_idx

            print((label_assignment.sum(1) == 0).sum())

            sbd_proposals['indexes'].append(img_id)
            sbd_proposals['mat'].append(label_assignment)

        print(f'\rImage Index: {(index + 1):.0f}/{len(imgIds):.0f}  ', end='')

    pickle.dump(sbd_proposals, open(
        os.path.join("/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/data/trash", useless_file.format(lzc_idx)), 'wb'),
                pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")

    # ["voc", "coco"]
    # dataset = "voc"
    dataset = "coco"

    with open("/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/configs/file_path/file_paths.yaml") as f:
        yml_par = yaml.load(f)

    label_file = yml_par[dataset]["train_json_file"]
    if dataset == "voc":
        assignment_label_file = "/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/data/label_assign/voc_2012_prmref_assign_bg_filter_0.pkl"
    else:
        assignment_label_file = "/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/data/label_assign/coco_2017_assign_bg_filter_0.pkl"
    prm_model_path = yml_par[dataset]["prm_path"]

    print(label_file)
    print(assignment_label_file)
    print(prm_model_path)

    ###
    worker = 8

    cocoGt = COCO(label_file)
    imgIds = sorted(cocoGt.getImgIds())

    per_len = int(len(imgIds) / worker)

    n_gpus = torch.cuda.device_count()

    if dataset == "voc":
        generate_assign_function = generate_assign_voc2012
        backbone = fc_resnet50(num_classes=20)
        model = peak_response_mapping(backbone=backbone, sub_pixel_locating_factor=8)

        pretrained = torch.load(prm_model_path, map_location=torch.device('cpu'))
        try:
            pretrained = pretrained['model']
        except:
            pass
        model.load_state_dict(pretrained)
    else:
        generate_assign_function = generate_assugn_coco2017
        backbone = fc_resnet50(num_classes=80)
        model = peak_response_mapping(backbone=backbone, sub_pixel_locating_factor=8)
        pretrained = torch.load(prm_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(pretrained)

    jobs = []
    for lzc_idx in range(worker):
        if lzc_idx + 1 != worker:
            p = multiprocessing.Process(target=generate_assign_function,
                                        args=(imgIds[lzc_idx * per_len:(lzc_idx + 1) * per_len], lzc_idx,
                                              copy.deepcopy(model), "cuda:{}".format(int(lzc_idx % n_gpus)),
                                              yml_par, dataset, cocoGt))
        else:
            p = multiprocessing.Process(target=generate_assign_function,
                                        args=(imgIds[lzc_idx * per_len:], lzc_idx, copy.deepcopy(model),
                                              "cuda:{}".format(int(lzc_idx % n_gpus)),
                                              yml_par, dataset, cocoGt))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()

    res = dict(indexes=[], mat=[], score=[])
    lzc_idx = 0
    while (lzc_idx != worker):
        path = os.path.join("/home/lzc/WSIS-Benchmark/code/WSRCNN-troch1.6/data/trash", useless_file.format(lzc_idx))
        try:
            result = pickle.load(open(path, 'rb'))
            res['indexes'] += result['indexes']
            res['mat'] += result['mat']
            os.remove(path)

            lzc_idx += 1
        except:
            pass

    print("imgs len: " + str(len(imgIds)))
    print("indexes len: " + str(len(res["indexes"])))
    print("mat len: " + str(len(res["mat"])))

    pickle.dump(res, open(assignment_label_file, 'wb'), pickle.HIGHEST_PROTOCOL)


