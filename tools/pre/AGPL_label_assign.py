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

def assign_coco2017(imgIds, worker_id, model, device, dataset, cocoGt):
    img_dir = "./data/coco2017/train2017"
    cob_original_file = "./data/coco2017/COB-COCO"
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

            peak_list = valid_peak_list.cpu().numpy()

            bg_ind_agg = np.zeros(num_proposal,dtype=np.float32)
            for j in range(len(peak_response_maps)):

                class_idx = peak_list[j, 1]
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

            bg_ind_agg = (bg_ind_agg != 0).astype(np.float32) * (label_assignment.sum(1) == 0).astype(np.float32)
            label_assignment[bg_ind_agg != 0, 0] = cluster_idx
            sbd_proposals['indexes'].append(img_id)
            sbd_proposals['mat'].append(label_assignment)

    pickle.dump(sbd_proposals, open(
        os.path.join(trash, useless_file.format(worker_id)), 'wb'),
                pickle.HIGHEST_PROTOCOL)


def assign_voc2012(imgIds, worker_id, model, device, dataset, cocoGt):
    img_dir = "./data/VOC2012/JPEGImages"
    cob_original_file = "./data/VOC2012/COB_SBD_trainaug"
    model = model.inference().to(device)
    sbd_proposals = dict(indexes=[], mat=[], score=[])
    for index in tqdm(range(len(imgIds))):
        img_id = imgIds[index]
        cluster_idx = 1

        # img
        s = str(int(img_id))
        file_name = s[:4] + '_' + s[4:]
        img = Image.open(os.path.join(img_dir, file_name + '.jpg')).convert('RGB')
        inputs = open_transform(img).unsqueeze(0).to(device)

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
        label_assignment = np.zeros((num_proposal, 21), dtype=np.float32)

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


    pickle.dump(sbd_proposals, open(
        os.path.join(trash, useless_file.format(worker_id)), 'wb'),
                pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = parse_args()

    multiprocessing.set_start_method("spawn")

    dataset = args.dataset

    if dataset == "voc":
        label_file = "./data/VOC2012/annotations/voc_2012_trainaug.json"
        assignment_label_file = "./data/label_assign/voc_2012_label_assign.pkl"
        prm_model_path = "./data/model_weight/prm_voc.pth"
    elif dataset == "coco":
        label_file = "./data/coco2017/annotations/instances_train2017.json"
        assignment_label_file = "./data/label_assign/coco_2017_label_assign.pkl"
        prm_model_path = "./data/model_weight/prm_coco.pth"
    else:
        raise NotImplementedError

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
        assign_fun = assign_voc2012
        backbone = fc_resnet50(num_classes=20)
        model = peak_response_mapping(backbone=backbone, sub_pixel_locating_factor=8)

        pretrained = torch.load(prm_model_path, map_location=torch.device('cpu'))
        try:
            pretrained = pretrained['model']
        except:
            pass
        model.load_state_dict(pretrained)
    else:
        assign_fun = assign_coco2017
        backbone = fc_resnet50(num_classes=80)
        model = peak_response_mapping(backbone=backbone, sub_pixel_locating_factor=8)
        pretrained = torch.load(prm_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(pretrained)

    jobs = []
    for worker_id in range(worker):
        if worker_id + 1 != worker:
            p = multiprocessing.Process(target=assign_fun,
                                        args=(imgIds[worker_id * per_len:(worker_id + 1) * per_len], worker_id,
                                              copy.deepcopy(model), "cuda:{}".format(int(worker_id % n_gpus)),
                                              dataset, cocoGt))
        else:
            p = multiprocessing.Process(target=assign_fun,
                                        args=(imgIds[worker_id * per_len:], worker_id, copy.deepcopy(model),
                                              "cuda:{}".format(int(worker_id % n_gpus)),
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


