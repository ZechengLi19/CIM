import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils
import numpy as np
from torchvision.ops import box_iou,nms
from torch_geometric.nn import GCNConv, GATConv, APPNP
from torch_geometric.nn import Sequential as GNN_Sequential
from utils.mask_utils import mask_iou
from modeling.mmcv_box.visualization.image import imshow_det_bboxes
from modeling.pamr import PAMR
import torchvision

import cv2
from PIL import Image
import os
from scipy.io import loadmat
import json
from matplotlib import pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
import time
from scipy.ndimage import zoom

l1loss = nn.L1Loss()
mseloss = nn.MSELoss()
agg_loss = nn.BCELoss()

def crf_inference_label(img, labels, t=1, n_labels=21, gt_prob=0.7):
    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)

def id_2_clsname(annotation_file_path):
    with open(annotation_file_path,"r") as f:
        content = f.readlines()
    json_dict = json.loads(content[0])
    cls_name_map = [cat["name"] for cat in json_dict["categories"]]
    cls_id_map = [cat["id"] for cat in json_dict["categories"]]

    return cls_name_map,cls_id_map

def two_BCEloss(predict_cls, mat, labels):
    label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
    label_tmp[:, 1:] = labels
    aggregation = predict_cls.max(0)[0]
    loss2 = mil_losses(aggregation, label_tmp)  # bag loss
    ind = (mat != 0).sum(1) != 0
    refine_tmp = predict_cls[ind, :]
    gt_tmp = (mat[ind, :] != 0).float()
    class_num = labels.shape[1] + 1
    if len(gt_tmp) != 0:
        loss1 = mil_losses(refine_tmp, gt_tmp)  # instance loss
    else:
        loss1 = torch.tensor(0.).to(device=loss2.device)
    return loss1 * class_num, loss2

def graph_two_Loss_mean(predict_cls, mat, labels):
    label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
    label_tmp[:, 1:] = labels
    aggregation = predict_cls.max(0)[0]
    loss2 = mil_losses(aggregation, label_tmp)

    loss1 = torch.tensor(0.).cuda(device=loss2.device)

    bg_ind = np.setdiff1d(mat[:, 0].cpu().numpy(), [0])
    if len(bg_ind) == 0:
        # no bg
        bg_ind = 10000
    else:
        assert len(bg_ind) == 1
        bg_ind = bg_ind[0]
    # bg_ind = gt_assignment.max().item()
    fg_bg_num = 1e-6
    for cluster_ind in mat.unique():
        if cluster_ind.item() != 0 and cluster_ind.item() != bg_ind:
            TFmat = (mat == cluster_ind)
            refine_tmp = predict_cls[TFmat.sum(1) != 0, :]
            col_ind = (TFmat.sum(0) != 0).float()
            refine_tmp_vector = refine_tmp.mean(0)
            fg_bg_num += refine_tmp.shape[0]
            loss1 += refine_tmp.shape[0] * mil_losses(refine_tmp_vector, col_ind)

        elif cluster_ind.item() == bg_ind:
            TFmat = (mat == cluster_ind)
            refine_tmp = predict_cls[TFmat.sum(1) != 0, :]
            gt_tmp = (mat[TFmat.sum(1) != 0, :] != 0).float()
            fg_bg_num += refine_tmp.shape[0]
            loss1 += refine_tmp.shape[0] * mil_losses(refine_tmp, gt_tmp)

    loss1 = loss1 / fg_bg_num
    # return 4 * loss1, loss2
    return 3 * 4 * loss1, loss2
    # ban
    # return 0.6 * labels.shape[-1] * loss1, loss2
    # return 1. * labels.shape[-1] * loss1, loss2


# WSOD part
# mil bag loss
def mil_bag_loss(predict_cls, predict_det,labels):
    pred = predict_cls * predict_det
    pred = torch.sum(pred,dim=0,keepdim=True)
    pred = pred.clamp(1e-6, 1 - 1e-6)

    # bg in pred
    if pred.shape[-1]-1 == labels.shape[-1]:
        label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
        label_tmp[:, 1:] = labels

    # bg not in pred
    else:
        label_tmp = labels.new_ones(labels.shape[0], labels.shape[1])
        label_tmp[:, 0:] = labels

    loss = - (label_tmp * torch.log(pred) + (1 - label_tmp) * torch.log(1 - pred)) # BCE loss

    return loss.mean()

def mil_bag_loss_ab3(predict_cls, predict_det,labels):
    pred = predict_cls * predict_det
    pred = torch.sum(pred[:,1:],dim=0,keepdim=True)
    pred = pred.clamp(1e-6, 1 - 1e-6)

    loss = - (labels * torch.log(pred) + (1 - labels) * torch.log(1 - pred)) # BCE loss

    return loss.mean()

# loss_weight
# 在 refine 分支中使用的 bag loss
def loss_weight_bag_loss(predict,pseudo_labels,label_tmp,loss_weight):
    assert predict.ndim == 2
    label_tmp = label_tmp.squeeze()
    assert label_tmp.ndim == 1

    ind = (pseudo_labels != 0).sum(-1) != 0
    tmp_pseudo_label = (pseudo_labels != 0).float()
    assert tmp_pseudo_label.max() == 1

    # label == 1 的max
    ind_agg_value, ind_agg_index = torch.max(ind[:,None] * predict * tmp_pseudo_label,dim=0)
    # label == 0，1 的max
    agg_value, agg_index = torch.max(predict,dim=0)

    # 将 label == 1 的max强制收到 aggression 中
    # 使用 label == 0，1 的 max 填充 label == 0 的部分
    aggression = (ind_agg_value * label_tmp) + (agg_value * (1 - label_tmp))
    aggression = aggression.clamp(1e-6, 1 - 1e-6)

    # 将 loss_weight 取出来
    label_flag = label_tmp == 1
    aggression_index = torch.zeros_like(agg_index)
    aggression_index[label_flag] = ind_agg_index[label_flag]
    aggression_index[~label_flag] = agg_index[~label_flag]

    label_weight = loss_weight[aggression_index]
    label_weight[~label_flag] = 1
    # label_weight[~label_flag] = torch.max(loss_weight)

    loss = - (label_tmp * torch.log(aggression) + (1 - label_tmp) * torch.log(1 - aggression)) * label_weight # BCE loss

    return loss.mean()



# cal cls_loss, iou_loss
# use image label
def cal_cls_iou_loss_function_full(cls_score, iou_score, pseudo_labels, pseudo_iou_label,loss_weights, labels, del_iou_branch=False):
    pseudo_iou_label = pseudo_iou_label.flatten()
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
    iou_score = iou_score.clamp(1e-6, 1 - 1e-6)

    label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
    label_tmp[:, 1:] = labels

    ind = (pseudo_labels != 0).sum(-1) != 0

    # aggregation = cls_score.max(0)[0]
    # bag_loss = mil_losses(aggregation, label_tmp)

    if del_iou_branch:
        bag_loss = loss_weight_bag_loss(cls_score, pseudo_labels, label_tmp, loss_weights)

    else:
        if iou_score.shape[-1] == 1:
            # 对于背景类，并没有完整与否的语意
            temp_op_score = torch.concat((cls_score[:,0:1], cls_score[:,1:] * iou_score),dim=1)
            bag_loss = loss_weight_bag_loss(temp_op_score, pseudo_labels, label_tmp, loss_weights)
        else:
            # bag loss 的含义为： 有/无一个完整的instance
            # 效果最好
            bag_loss = loss_weight_bag_loss(cls_score*iou_score, pseudo_labels, label_tmp, loss_weights)

            # # bag loss 的含义为： 对于有标签的类别，有一个完整的instance，但对于一个没有标签的类别，没有这个cls
            # bag_mask = torch.zeros(labels.shape[0], labels.shape[1] + 1,device=labels.device)
            # bag_mask[:, 1:] = labels
            #
            # bag_loss = loss_weight_bag_loss(bag_mask * cls_score * iou_score + (1 - bag_mask) * cls_score,
            #                                 pseudo_labels, label_tmp, loss_weights)

        # bag_loss = loss_weight_bag_loss(cls_score, pseudo_labels, labels, loss_weights)


    ind_cls_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    ind_iou_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    f_ind_cls_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    f_ind_iou_loss = torch.tensor(0.).to(device=pseudo_labels.device)

    # # all ~label
    # temp_cls_loss = -(1 - label_tmp) * torch.log(1 - cls_score)
    # f_ind_cls_loss = temp_cls_loss.sum() / (1 - label_tmp).sum() / cls_score.shape[0]
    #
    # fake_iou_label = torch.zeros_like(iou_score,device=pseudo_labels.device)
    # f_ind_iou_loss = nn.functional.smooth_l1_loss((1 - label_tmp) * iou_score, fake_iou_label, reduction="none")
    # f_ind_iou_loss = f_ind_iou_loss.sum() / (1 - label_tmp).sum() / cls_score.shape[0]

    if ind.sum() != 0:
        pseudo_labels = (pseudo_labels[ind] != 0).float()
        assert pseudo_labels.max() == 1
        pseudo_iou_label = pseudo_iou_label[ind]
        cls_score = cls_score[ind]
        iou_score = iou_score[ind]
        loss_weights = loss_weights[ind]

        # cls_loss
        # ce loss
        ind_cls_loss = -pseudo_labels * torch.log(cls_score) * loss_weights.view(-1,1)
        ind_cls_loss = ind_cls_loss.sum() / pseudo_labels.sum()

        # # bce loss 掉点
        # ind_cls_loss = -(pseudo_labels * torch.log(cls_score) + (1 - pseudo_labels) * torch.log(1 - cls_score)) * loss_weights.view(-1,1)
        # ind_cls_loss = ind_cls_loss.mean() * cls_score.shape[-1]

        fg_ind = (pseudo_labels[:,1:] != 0).sum(-1) != 0
        # bg_ind = ~fg_ind
        # pseudo_iou_label[bg_ind] = 0
        if fg_ind.sum() != 0:
            fg_pseudo_labels = pseudo_labels[fg_ind]
            fg_pseudo_iou_label = pseudo_iou_label[fg_ind]
            fg_iou_score = iou_score[fg_ind]
            fg_loss_weights = loss_weights[fg_ind]

            # iou score 与类别相关
            # 压成1D
            if fg_iou_score.shape[-1] == fg_pseudo_labels.shape[-1]:
                fg_iou_score = (fg_pseudo_labels * fg_iou_score).sum(-1)
            # iou score 与类别无关
            elif fg_iou_score.shape[-1] == 1:
                fg_iou_score = fg_iou_score.squeeze()
            else:
                raise AssertionError

            # ind_iou_loss = - (fg_pseudo_iou_label * torch.log(fg_iou_score) + (1 - fg_pseudo_iou_label) * torch.log(1 - fg_iou_score)) * fg_loss_weights
            # ind_iou_loss = ind_iou_loss.sum() / fg_pseudo_labels.sum()

            ind_iou_loss = nn.functional.smooth_l1_loss(fg_iou_score, fg_pseudo_iou_label,reduction="none") * fg_loss_weights
            ind_iou_loss = ind_iou_loss.sum() / fg_pseudo_labels.sum()

        else:
            ind_iou_loss = torch.tensor(0.).to(device=pseudo_labels.device)

    return ind_cls_loss, ind_iou_loss, f_ind_cls_loss, f_ind_iou_loss, bag_loss

# for DC --> iou_label is a matrix
def cal_cls_iou_loss_function_full_DC(cls_score, iou_score, pseudo_labels, pseudo_iou_label, group_iou_relationship,loss_weights, labels, del_iou_branch=False):
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
    iou_score = iou_score.clamp(1e-6, 1 - 1e-6)

    label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
    label_tmp[:, 1:] = labels

    ind = (pseudo_labels != 0).sum(-1) != 0

    if del_iou_branch:
        bag_loss = loss_weight_bag_loss(cls_score, pseudo_labels, label_tmp, loss_weights)

    else:
        if iou_score.shape[-1] == 1:
            # 对于背景类，并没有完整与否的语意
            temp_op_score = torch.concat((cls_score[:,0:1], cls_score[:,1:] * iou_score),dim=1)
            bag_loss = loss_weight_bag_loss(temp_op_score, pseudo_labels, label_tmp, loss_weights)
        else:
            # bag loss 的含义为： 有/无一个完整的instance
            # 效果最好
            bag_loss = loss_weight_bag_loss(cls_score*iou_score, pseudo_labels, label_tmp, loss_weights)

    ind_cls_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    ind_iou_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    f_ind_cls_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    f_ind_iou_loss = torch.tensor(0.).to(device=pseudo_labels.device)

    if ind.sum() != 0:
        pseudo_labels = (pseudo_labels[ind] != 0).float()
        assert pseudo_labels.max() == 1
        pseudo_iou_label = pseudo_iou_label[ind]
        group_iou_relationship = group_iou_relationship[ind]
        cls_score = cls_score[ind]
        iou_score = iou_score[ind]
        loss_weights = loss_weights[ind]

        # cls_loss
        # ce loss
        ind_cls_loss = -pseudo_labels * torch.log(cls_score) * loss_weights.view(-1,1)
        ind_cls_loss = ind_cls_loss.sum() / pseudo_labels.sum()

        ind_iou_loss = group_iou_relationship * nn.functional.smooth_l1_loss(iou_score, pseudo_iou_label,reduction="none") * loss_weights.view(-1,1)
        ind_iou_loss = ind_iou_loss.sum() / group_iou_relationship.sum()

    return ind_cls_loss, ind_iou_loss, f_ind_cls_loss, f_ind_iou_loss, bag_loss


# use group size to normalize num
def cal_cls_iou_loss_function_full_group_normalize(cls_score, iou_score, pseudo_labels, pseudo_iou_label,loss_weights, labels, group_assign, del_iou_branch=False):
    pseudo_iou_label = pseudo_iou_label.flatten()
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
    iou_score = iou_score.clamp(1e-6, 1 - 1e-6)

    label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
    label_tmp[:, 1:] = labels

    ind = (pseudo_labels != 0).sum(-1) != 0

    # aggregation = cls_score.max(0)[0]
    # bag_loss = mil_losses(aggregation, label_tmp)

    if del_iou_branch:
        bag_loss = loss_weight_bag_loss(cls_score, pseudo_labels, label_tmp, loss_weights)

    else:
        if iou_score.shape[-1] == 1:
            # 对于背景类，并没有完整与否的语意
            temp_op_score = torch.concat((cls_score[:,0:1], cls_score[:,1:] * iou_score),dim=1)
            bag_loss = loss_weight_bag_loss(temp_op_score, pseudo_labels, label_tmp, loss_weights)
        else:
            # bag loss 的含义为： 有/无一个完整的instance
            # 效果最好
            bag_loss = loss_weight_bag_loss(cls_score*iou_score, pseudo_labels, label_tmp, loss_weights)

    ind_cls_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    ind_iou_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    f_ind_cls_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    f_ind_iou_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    bg_ind_cls_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    fg_ind_cls_loss = torch.tensor(0.).to(device=pseudo_labels.device)

    if ind.sum() != 0:
        pseudo_labels = (pseudo_labels[ind] != 0).float()
        assert pseudo_labels.max() == 1
        pseudo_iou_label = pseudo_iou_label[ind]
        cls_score = cls_score[ind]
        iou_score = iou_score[ind]
        loss_weights = loss_weights[ind]
        group_assign = group_assign[ind]

        assert group_assign.ndim == 2
        group_ids = torch.unique(group_assign)
        cls_num = 1e-5
        fg_num = 1e-5
        bg_num = 1e-5
        for id in group_ids:
            group_flag = group_assign == id  # N * 21
            group_flag = group_flag.sum(1) != 0

            if id == -2 or id == 0:
                continue
            else:
                if id == -1:
                    bg_ind_cls_loss = (-torch.tensor(group_flag[:, None], dtype=torch.float32) * pseudo_labels * torch.log(cls_score) * loss_weights.view(-1, 1)).sum()
                    bg_num = group_flag.sum()

                else:
                    cls_num += 1

                    fg_num += group_flag.sum()
                    fg_ind_cls_loss = fg_ind_cls_loss + (
                                -torch.tensor(group_flag[:, None], dtype=torch.float32) * pseudo_labels * torch.log(
                                cls_score) * loss_weights.view(-1, 1)).sum() / group_flag.sum()
                    fg_ind = (torch.sum(pseudo_labels[:, 1:] != 0,keepdim=False,dim=-1) * torch.tensor(group_flag,dtype=torch.float32)) != 0
                    if fg_ind.sum() != 0:
                        fg_pseudo_labels = pseudo_labels[fg_ind]
                        fg_pseudo_iou_label = pseudo_iou_label[fg_ind]
                        fg_iou_score = iou_score[fg_ind]
                        fg_loss_weights = loss_weights[fg_ind]

                        # iou score 与类别相关
                        # 压成1D
                        if fg_iou_score.shape[-1] == fg_pseudo_labels.shape[-1]:
                            fg_iou_score = (fg_pseudo_labels * fg_iou_score).sum(-1)
                        # iou score 与类别无关
                        elif fg_iou_score.shape[-1] == 1:
                            fg_iou_score = fg_iou_score.squeeze()
                        else:
                            raise AssertionError

                        ind_iou_loss = ind_iou_loss + \
                                       (nn.functional.smooth_l1_loss(fg_iou_score, fg_pseudo_iou_label,reduction="none")
                                        * fg_loss_weights).sum() / fg_ind.sum()

        ind_cls_loss = (bg_ind_cls_loss + fg_num * fg_ind_cls_loss / cls_num) / (bg_num + fg_num)
        ind_iou_loss = ind_iou_loss / cls_num

    return ind_cls_loss, ind_iou_loss, f_ind_cls_loss, f_ind_iou_loss, bag_loss

def cal_cls_iou_loss_function_full_group_assign_mean(cls_score, iou_score, pseudo_labels, pseudo_iou_label,loss_weights, labels, group_assign):
    pseudo_iou_label = pseudo_iou_label.flatten()
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
    iou_score = iou_score.clamp(1e-6, 1 - 1e-6)

    label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
    label_tmp[:, 1:] = labels

    if iou_score.shape[-1] == 1:
        # 对于背景类，并没有完整与否的语意
        temp_op_score = torch.concat((cls_score[:,0:1], cls_score[:,1:] * iou_score),dim=1)
        bag_loss = loss_weight_bag_loss(temp_op_score, pseudo_labels, label_tmp, loss_weights)
    else:
        # bag loss 的含义为： 有/无一个完整的instance
        # 效果最好
        bag_loss = loss_weight_bag_loss(cls_score*iou_score, pseudo_labels, label_tmp, loss_weights)
    # bag_loss = loss_weight_bag_loss(cls_score*iou_score, pseudo_labels, label_tmp, loss_weights)

    ind_cls_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    ind_iou_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    f_ind_cls_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    f_ind_iou_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    ind_cls_loss_num = 1e-6
    ind_iou_loss_num = 1e-6

    # iou score 与类别相关
    # 压成1D
    if iou_score.shape[-1] == pseudo_labels.shape[-1]:
        iou_score = (pseudo_labels * iou_score).sum(-1)
    # iou score 与类别无关
    elif iou_score.shape[-1] == 1:
        iou_score = iou_score.squeeze()
    else:
        raise AssertionError

    assert group_assign.ndim == 2
    group_ids = torch.unique(group_assign)
    for id in group_ids:
        group_flag = group_assign == id # N * 21
        group_flag = group_flag.sum(1) != 0

        if id == -2 or id == 0:
            continue
        elif id == -1:
            ind_cls_loss += -(group_flag[:,None] * pseudo_labels * torch.log(cls_score) * loss_weights.view(-1,1)).sum()
            ind_cls_loss_num += group_flag.sum()

        else:
            sub_bag_label = pseudo_labels[group_flag][0][None,:]
            sub_bag_loss_weight = loss_weights[group_flag][0]
            mean_cls_score = torch.mean(cls_score[group_flag], dim=0, keepdim=True)

            ind_cls_loss += -(sub_bag_label * torch.log(mean_cls_score) * sub_bag_loss_weight).sum() * group_flag.sum()
            ind_cls_loss_num += group_flag.sum()

            # 模仿上面的设计
            # bg 全算
            fg_iou_score = iou_score[group_flag]
            fg_pseudo_iou_label = pseudo_iou_label[group_flag]

            ind_iou_loss_num += group_flag.sum()

            iou_0 = fg_pseudo_iou_label == 0
            iou_1 = fg_pseudo_iou_label == 1

            if iou_0.sum() != 0:
                fg_iou_score_0 = fg_iou_score[iou_0]
                ind_iou_loss += (nn.functional.smooth_l1_loss(fg_iou_score_0, torch.zeros_like(fg_iou_score_0,device=fg_iou_score_0.device),reduction="none") * sub_bag_loss_weight).sum()

            if iou_1.sum() != 0:
                fg_iou_score_1 = fg_iou_score[iou_1]
                mean_fg_iou_score_1 = torch.mean(fg_iou_score_1)

                ind_iou_loss += (nn.functional.smooth_l1_loss(mean_fg_iou_score_1, torch.tensor(1.,device=fg_iou_score_1.device),reduction="none") * sub_bag_loss_weight).sum() * iou_1.sum()

    ind_cls_loss = ind_cls_loss / ind_cls_loss_num
    ind_iou_loss = ind_iou_loss / ind_iou_loss_num

    return ind_cls_loss, ind_iou_loss, f_ind_cls_loss, f_ind_iou_loss, bag_loss

class PeakCluster(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.peak_cluster = nn.Linear(dim_in, dim_out)
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.peak_cluster.weight, std=0.01)
        init.constant_(self.peak_cluster.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'peak_cluster.weight': 'peak_cluster_w',
            'peak_cluster.bias': 'peak_cluster_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        cluster_score = self.peak_cluster(x)
        cluster_score = F.softmax(cluster_score, dim=1)
        return cluster_score

class cls_iou_model(nn.Module):
    def __init__(self, dim_in, dim_out,ref_num,class_agnostic=False):
        super(cls_iou_model, self).__init__()

        # WSDDN mode
        # 包括bg
        self.classifier = nn.Linear(dim_in,dim_out)
        self.inner_detection = nn.Linear(dim_in,dim_out)
        self.outer_detection = nn.Linear(dim_in,dim_out)

        # # 不包括bg
        # self.classifier = nn.Linear(dim_in,dim_out - 1)
        # if class_agnostic:
        #     self.detection = nn.Linear(dim_in, 1)
        #     print("class_agnostic")
        # else:
        #     self.detection = nn.Linear(dim_in, dim_out - 1)
        #     print("not class_agnostic")
        ################

        # self.classifier = nn.Linear(dim_in,dim_out)
        # self.detection = nn.Linear(dim_in,dim_out)

        self.ref_cls = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(ref_num)])

        # bg class learn the iou score
        self.ref_iou = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(ref_num)])

    def detectron_weight_mapping(self):
        detectron_weight_mapping = dict()
        for name, _ in self.named_parameters():
            detectron_weight_mapping[name] = name

        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, seg_feature,out_seg_feature, diff_feature):
        if seg_feature.dim() == 4:
            seg_feature = seg_feature.squeeze(3).squeeze(2)
            out_seg_feature = out_seg_feature.squeeze(3).squeeze(2)
            diff_feature = diff_feature.squeeze(3).squeeze(2)

        # WSDDN
        predict_cls = self.classifier(seg_feature)
        predict_cls = nn.functional.softmax(predict_cls,dim=-1)

        inner_predict_det = self.inner_detection(seg_feature)
        # outer_predict_det = self.outer_detection(out_seg_feature)

        ### add model
        # predict_det = inner_predict_det + outer_predict_det
        predict_det = inner_predict_det

        predict_det = nn.functional.softmax(predict_det,dim=0)
        ############

        # # iou label guide
        # predict_cls = self.classifier(seg_feat)
        # predict_cls = nn.functional.softmax(predict_cls, dim=-1)
        #
        # predict_det = self.detection(diff_feature)
        # predict_det = nn.functional.sigmoid(predict_det)

        ref_cls_score = []
        ref_iou_score = []

        for i, (cls_layer, iou_layer) in enumerate(zip(self.ref_cls,self.ref_iou)):
            cls_score = cls_layer(seg_feature)
            cls_score = nn.functional.softmax(cls_score, dim=-1)
            ref_cls_score.append(cls_score)

            # iou_score = iou_layer(diff_feature)
            iou_score = iou_layer(seg_feature)
            iou_score = F.sigmoid(iou_score)
            ref_iou_score.append(iou_score)

        return predict_cls, predict_det, ref_cls_score, ref_iou_score

# all WSDDN head
class cls_iou_model_wsddn(nn.Module):
    def __init__(self, dim_in, dim_out,ref_num,class_agnostic=False):
        super(cls_iou_model_wsddn, self).__init__()

        # WSDDN mode
        # 包括bg
        self.classifier = nn.Linear(dim_in,dim_out)
        self.inner_detection = nn.Linear(dim_in,dim_out)
        self.outer_detection = nn.Linear(dim_in,dim_out)

        self.ref_cls = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(ref_num)])

        # bg class learn the iou score
        self.ref_iou = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(ref_num)])

    def detectron_weight_mapping(self):
        detectron_weight_mapping = dict()
        for name, _ in self.named_parameters():
            detectron_weight_mapping[name] = name

        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, seg_feature,out_seg_feature, diff_feature):
        if seg_feature.dim() == 4:
            seg_feature = seg_feature.squeeze(3).squeeze(2)
            out_seg_feature = out_seg_feature.squeeze(3).squeeze(2)
            diff_feature = diff_feature.squeeze(3).squeeze(2)

        # WSDDN
        predict_cls = self.classifier(seg_feature)
        predict_cls = nn.functional.softmax(predict_cls,dim=-1)

        predict_det = self.inner_detection(seg_feature)
        predict_det = nn.functional.softmax(predict_det,dim=0)

        ref_cls_score = []
        ref_iou_score = []

        for i, (cls_layer, iou_layer) in enumerate(zip(self.ref_cls,self.ref_iou)):
            cls_score = cls_layer(seg_feature)
            cls_score = nn.functional.softmax(cls_score, dim=-1)
            ref_cls_score.append(cls_score)

            iou_score = iou_layer(seg_feature)
            iou_score = nn.functional.softmax(iou_score,dim=0)
            ref_iou_score.append(iou_score)

        return predict_cls, predict_det, ref_cls_score, ref_iou_score

# WSDDN head 不包括 bg
class cls_iou_model_ab3(nn.Module):
    def __init__(self, dim_in, dim_out,ref_num,class_agnostic=False):
        super(cls_iou_model_ab3, self).__init__()

        # WSDDN mode
        # 包括bg
        self.classifier = nn.Linear(dim_in,dim_out-1)
        self.inner_detection = nn.Linear(dim_in,dim_out-1)

        # # 不包括bg
        # self.classifier = nn.Linear(dim_in,dim_out - 1)
        # if class_agnostic:
        #     self.detection = nn.Linear(dim_in, 1)
        #     print("class_agnostic")
        # else:
        #     self.detection = nn.Linear(dim_in, dim_out - 1)
        #     print("not class_agnostic")
        ################

        # self.classifier = nn.Linear(dim_in,dim_out)
        # self.detection = nn.Linear(dim_in,dim_out)

        self.ref_cls = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(ref_num)])

        # bg class learn the iou score
        self.ref_iou = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(ref_num)])

    def detectron_weight_mapping(self):
        detectron_weight_mapping = dict()
        for name, _ in self.named_parameters():
            detectron_weight_mapping[name] = name

        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, seg_feature,out_seg_feature, diff_feature):
        if seg_feature.dim() == 4:
            seg_feature = seg_feature.squeeze(3).squeeze(2)
            out_seg_feature = out_seg_feature.squeeze(3).squeeze(2)
            diff_feature = diff_feature.squeeze(3).squeeze(2)

        # WSDDN
        predict_cls = self.classifier(seg_feature)
        predict_cls = nn.functional.softmax(predict_cls,dim=-1)
        predict_cls = torch.concat((torch.zeros(predict_cls.shape[0],1,device=predict_cls.device),predict_cls),dim=1)

        predict_det = self.inner_detection(seg_feature)

        predict_det = nn.functional.softmax(predict_det,dim=0)
        predict_det = torch.concat((torch.zeros(predict_cls.shape[0],1,device=predict_cls.device),predict_det),dim=1)
        ############

        # # iou label guide
        # predict_cls = self.classifier(seg_feat)
        # predict_cls = nn.functional.softmax(predict_cls, dim=-1)
        #
        # predict_det = self.detection(diff_feature)
        # predict_det = nn.functional.sigmoid(predict_det)

        ref_cls_score = []
        ref_iou_score = []

        for i, (cls_layer, iou_layer) in enumerate(zip(self.ref_cls,self.ref_iou)):
            cls_score = cls_layer(seg_feature)
            cls_score = nn.functional.softmax(cls_score, dim=-1)
            ref_cls_score.append(cls_score)

            # iou_score = iou_layer(diff_feature)
            iou_score = iou_layer(seg_feature)
            iou_score = F.sigmoid(iou_score)
            ref_iou_score.append(iou_score)

        return predict_cls, predict_det, ref_cls_score, ref_iou_score

class mist_layer(nn.Module):
    def __init__(self, portion=0.1, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85,sample=False,test_mode=False,get_diffuse_gt=False):
        super(mist_layer, self).__init__()
        self.portion = portion
        self.full_thr = full_thr
        self.iou_th = iou_thr
        self.asy_iou_th = asy_iou_th
        self.sample = sample
        # 当 test mode 生效时，只会 output gt 当标签，其他统一为 0
        # 用于对分布进行统计
        # 推荐对 val 进行统计
        self.test_mode = test_mode
        # False 指 得到原本的 gt
        # True 指 得到扩散以后的 gt
        self.get_diffuse_gt = get_diffuse_gt

        print("mist_layer--> portion:{}, full_thr: {}, iou_thr: {}".format(portion,full_thr,iou_thr))
        print("sample:{}, test_mode:{}, get_diffuse_gt: {}".format(sample, test_mode, get_diffuse_gt))

    # instance_list -> [{},{}...]
    # {} -> {score: float, mask_id: int}
    def instance_nms(self, instance_list, iou_map):
        instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)

        selected_instances_id = []
        while len(instance_list) > 0:
            src_instance = instance_list.pop(0)
            selected_instances_id.append(src_instance["mask_id"])

            src_mask_id = src_instance["mask_id"]

            def iou_filter(dst_instance):
                dst_mask_id = dst_instance["mask_id"]

                iou = iou_map[src_mask_id][dst_mask_id]
                if iou < self.iou_th:
                    return True
                else:
                    return False

            instance_list = list(filter(iou_filter, instance_list))

        return selected_instances_id

    # instance_list -> [{},{}...]
    # {} -> {score: float, mask_id: int}
    def instance_asy_nms(self, instance_list, asy_iou_map):
        instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)

        selected_instances_id = []
        while len(instance_list) > 0:
            src_instance = instance_list.pop(0)
            selected_instances_id.append(src_instance["mask_id"])

            src_mask_id = src_instance["mask_id"]

            def iou_filter(dst_instance):
                dst_mask_id = dst_instance["mask_id"]

                iou = asy_iou_map[src_mask_id][dst_mask_id]
                if iou < self.asy_iou_th:
                    return True
                else:
                    return False

            instance_list = list(filter(iou_filter, instance_list))

        return selected_instances_id

    @torch.no_grad()
    def mist_label(self, preds, rois, label, iou_map=None, asy_iou_map=None):
        if label.dim() != 1:
            label = label.squeeze()

        assert label.dim() == 1
        assert label.shape[-1] == 20 or label.shape[-1] == 80

        # bg remove
        preds = (preds if preds.shape[-1] == label.shape[-1] else preds[:, 1:]).clone()  # remove background class if present
        keep_count = int(np.ceil(self.portion * preds.shape[0]))
        klasses = label.nonzero(as_tuple=True)[0]
        # one hot label
        gt_labels = torch.zeros((preds.shape[0], label.shape[-1] + 1), dtype=preds.dtype, device=preds.device)
        gt_weights = -torch.ones((preds.shape[0],), dtype=preds.dtype, device=preds.device)

        for c in klasses:
            cls_prob_tmp = preds[:, c]

            keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals

            keep_rois = rois[keep_sort_idx]
            keep_cls_prob = cls_prob_tmp[keep_sort_idx]

            # iou nms
            if iou_map != None:
                temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]

                instance_list = []
                for i, prob in enumerate(keep_cls_prob):
                    instance = dict()

                    instance["score"] = prob
                    instance["mask_id"] = i
                    instance_list.append(instance)

                keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
                keep_nms_idx = torch.tensor(keep_nms_idx,device=preds.device)

            # box nms
            else:
                print("iou_map == None")
                keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)

            keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index

            is_higher_scoring_class = cls_prob_tmp[keep_nms_idx] > gt_weights[keep_nms_idx]
            keep_idxs = keep_nms_idx[is_higher_scoring_class]
            gt_labels[keep_idxs, :] = 0
            gt_labels[keep_idxs, c + 1] = 1
            gt_weights[keep_idxs] = cls_prob_tmp[keep_idxs]

        gt_idxs = torch.sum(gt_labels, dim=-1) > 0

        gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]

        return gt_boxes, gt_labels, gt_weights, gt_idxs

    @torch.no_grad()
    def mist_label_diffuse(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None):
        if label.dim() != 1:
            label = label.squeeze()

        assert label.dim() == 1
        assert label.shape[-1] == 20 or label.shape[-1] == 80

        # bg remove
        predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
        predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present

        preds = predict_cls * predict_det

        keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))
        klasses = label.nonzero(as_tuple=True)[0]
        # one hot label
        gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
        gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)

        # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
        asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]

        for c in klasses:
            cls_prob_tmp = predict_cls[:, c]
            if predict_det.shape[-1] == label.shape[-1]:
                det_prob_tmp = predict_det[:, c]
            elif predict_det.shape[-1] == 1:
                det_prob_tmp = predict_det[:, 0]
            else:
                raise AssertionError

            preds_tmp = preds[:, c]

            keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals

            keep_rois = rois[keep_sort_idx]
            keep_cls_prob = cls_prob_tmp[keep_sort_idx]

            # cal iou nms
            if iou_map != None:
                temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]

                instance_list = []
                for i, prob in enumerate(keep_cls_prob):
                    instance = dict()

                    instance["score"] = prob
                    instance["mask_id"] = i
                    instance_list.append(instance)

                keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
                keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)

            # box nms
            else:
                print("iou_map == None")
                keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)

            keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index

            assert asy_iou_map != None
            temp_asy_iou_map = asy_iou_map[:, keep_nms_idx]
            temp_asy_iou_map = temp_asy_iou_map > self.asy_iou_th

            flag = temp_asy_iou_map * asy_iou_flag
            if flag.sum() != 0:
                if self.test_mode and not self.get_diffuse_gt:
                    res_idx = keep_nms_idx[torch.sum(flag, dim=0) > 0]  # 原先的 res_idx --> 未diffuse

                else:
                    flag = flag[:, torch.sum(flag, dim=0) > 0]
                    res_det = flag * det_prob_tmp[:, None]
                    res_idx = torch.argmax(res_det, dim=0)  # 真正的 gt
                    res_idx = torch.unique(res_idx)

                is_higher_scoring_class = preds_tmp[res_idx] > gt_weights[res_idx]
                if is_higher_scoring_class.sum() > 0:
                    keep_idxs = res_idx[is_higher_scoring_class]
                    gt_labels[keep_idxs, :] = 0
                    gt_labels[keep_idxs, c + 1] = 1
                    gt_weights[keep_idxs] = preds_tmp[keep_idxs]

        gt_idxs = torch.sum(gt_labels, dim=-1) > 0
        # assert gt_idxs.sum() > 0

        gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]

        return gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag

    @torch.no_grad()
    def forward(self, predict_cls, predict_det, rois, labels, iou_map=None, asy_iou_map=None, diffuse = False):
        if rois.ndim == 3:
            rois = rois.squeeze(0)
        rois = rois[:,1:] # remove batch_id

        if diffuse:
            gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag = self.mist_label_diffuse(predict_cls, predict_det, rois, labels, iou_map, asy_iou_map)
            # gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag = self.mist_label_diffuse_reweight(predict_cls, predict_det, rois, labels, iou_map, asy_iou_map)
        else:
            if predict_det!= None:
                preds = predict_cls * predict_det
            else:
                preds = predict_cls

            gt_boxes, gt_labels, gt_weights, gt_idxs = self.mist_label(preds,rois,labels,iou_map,asy_iou_map)

        if gt_idxs.sum() == 0:
            return None, None, None, None

        if iou_map == None:
            overlaps = box_iou(rois, gt_boxes) # (proposal_num,gt_num)
        else:
            overlaps = iou_map[:, gt_idxs]

        max_overlap_v, max_overlap_idx = torch.max(overlaps,dim=-1)

        pseudo_labels = gt_labels[max_overlap_idx]
        loss_weights = gt_weights[max_overlap_idx]
        pseudo_iou_label = max_overlap_v

        if self.test_mode:
            bg_inds = (max_overlap_v != 1)
            pseudo_labels[bg_inds,:] = 0
            pseudo_labels[bg_inds,0] = 1
            pseudo_labels = pseudo_labels * loss_weights[:, None]

            return pseudo_labels, pseudo_iou_label, loss_weights, None

        else:
            ignore_inds = max_overlap_v == 0
            pseudo_labels[ignore_inds, :] = 0
            loss_weights[ignore_inds] = 0

            bg_inds = (max_overlap_v < self.iou_th) * ~ignore_inds
            pseudo_labels[bg_inds,:] = 0
            pseudo_labels[bg_inds,0] = 1

            try:
                # 将太大的proposal标记为bg
                big_proposal = ~asy_iou_flag
                pseudo_labels[big_proposal, :] = 0
                pseudo_labels[big_proposal, 0] = 1
            except:
                pass

            # 将 pseudo_iou_label 设置为离散值
            pseudo_iou_label[pseudo_iou_label > self.full_thr] = 1
            pseudo_iou_label[pseudo_iou_label <= self.full_thr] = 0

            # # 将 pseudo_iou_label 设置为连续值
            # pseudo_iou_label = (pseudo_iou_label - self.iou_th) / (1 - self.iou_th)

            group_assign = max_overlap_idx + 1 # 将 index 从 1 开始
            group_assign[bg_inds] = -1
            group_assign[ignore_inds] = -2
            # 所以最后的范围是 [-2, -1, , 1, 2, 3...]

            # transform pseudo_labels --> N * 21
            # 将 group_assign 变为 和 初始伪标签 mat 相同的形态
            group_assign = group_assign[:,None] * pseudo_labels

            return pseudo_labels, pseudo_iou_label, loss_weights, group_assign

# 在 argmax 时，使用 max --> cls * iou
# sample seed area and diffuse area
class mist_layer_v2(mist_layer):
    def __init__(self, portion=0.1, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85,sample=False,test_mode=False,get_diffuse_gt=False):
        super().__init__(portion, full_thr, iou_thr, asy_iou_th,sample,test_mode,get_diffuse_gt)
        print("mist_layer_v2")

    @torch.no_grad()
    def mist_label_diffuse(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None):
        if label.dim() != 1:
            label = label.squeeze()

        assert label.dim() == 1
        assert label.shape[-1] == 20 or label.shape[-1] == 80

        # bg remove
        predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
        predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present

        preds = predict_cls * predict_det

        keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))
        klasses = label.nonzero(as_tuple=True)[0]
        # one hot label
        gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
        gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)

        # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
        asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]

        for c in klasses:
            cls_prob_tmp = predict_cls[:, c]
            if predict_det.shape[-1] == label.shape[-1]:
                det_prob_tmp = predict_det[:, c]
            elif predict_det.shape[-1] == 1:
                det_prob_tmp = predict_det[:, 0]
            else:
                raise AssertionError

            preds_tmp = preds[:, c]

            keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals

            keep_rois = rois[keep_sort_idx]
            keep_cls_prob = cls_prob_tmp[keep_sort_idx]

            # cal iou nms
            if iou_map != None:
                temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]

                instance_list = []
                for i, prob in enumerate(keep_cls_prob):
                    instance = dict()

                    instance["score"] = prob
                    instance["mask_id"] = i
                    instance_list.append(instance)

                keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
                keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)

            # box nms
            else:
                print("iou_map == None")
                keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)

            keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index

            assert asy_iou_map != None
            temp_asy_iou_map = asy_iou_map[:, keep_nms_idx]
            temp_asy_iou_map = temp_asy_iou_map > self.asy_iou_th

            flag = temp_asy_iou_map * asy_iou_flag
            if flag.sum() != 0:
                if self.test_mode and not self.get_diffuse_gt:
                    res_idx = keep_nms_idx[torch.sum(flag, dim=0) > 0]  # 原先的 res_idx --> 未diffuse

                else:
                    flag = flag[:, torch.sum(flag, dim=0) > 0]
                    res_det = flag * det_prob_tmp[:, None]
                    res_idx = torch.argmax(res_det, dim=0)  # 真正的 gt
                    res_idx = torch.unique(res_idx)

                is_higher_scoring_class = preds_tmp[res_idx] > gt_weights[res_idx]
                if is_higher_scoring_class.sum() > 0:
                    keep_idxs = res_idx[is_higher_scoring_class]
                    gt_labels[keep_idxs, :] = 0
                    gt_labels[keep_idxs, c + 1] = 1
                    gt_weights[keep_idxs] = preds_tmp[keep_idxs]

        gt_idxs = torch.sum(gt_labels, dim=-1) > 0
        # assert gt_idxs.sum() > 0

        gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]

        return gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag

    @torch.no_grad()
    def forward(self, predict_cls, predict_det, rois, labels, iou_map=None, asy_iou_map=None, diffuse = False):
        if rois.ndim == 3:
            rois = rois.squeeze(0)
        rois = rois[:,1:] # remove batch_id

        if diffuse:
            gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag = self.mist_label_diffuse(predict_cls, predict_det, rois, labels, iou_map, asy_iou_map)
        else:
            if predict_det!= None:
                preds = predict_cls * predict_det
            else:
                preds = predict_cls

            gt_boxes, gt_labels, gt_weights, gt_idxs = self.mist_label(preds,rois,labels,iou_map,asy_iou_map)

        if gt_idxs.sum() == 0:
            return None, None, None, None

        if iou_map == None:
            overlaps = box_iou(rois, gt_boxes) # (proposal_num,gt_num)
        else:
            overlaps = iou_map[:, gt_idxs]

        if self.sample:
            # sample diffuse area
            if labels.dim() != 1:
                label = labels.squeeze()
            else:
                label = labels

            assert label.dim() == 1
            assert label.shape[-1] == 20 or label.shape[-1] == 80

            klasses = label.nonzero(as_tuple=True)[0]

            inds = torch.ones_like(gt_labels[:, 0], device=gt_labels.device)

            for c in klasses:
                class_idx = torch.nonzero(gt_labels[:, c + 1] == 1).flatten().cpu().numpy()
                if len(class_idx) == 0:
                    continue

                prob = gt_weights[class_idx].cpu().numpy()
                sampled_class_idx = np.random.choice(class_idx, size=len(class_idx), replace=True,
                                                     p=prob / prob.sum())
                sampled_class_idx = np.unique(sampled_class_idx)

                inds[class_idx] = 0
                inds[sampled_class_idx] = 1

            inds = inds == 1
            gt_weights = gt_weights[inds]
            gt_labels = gt_labels[inds, :]
            gt_boxes = gt_boxes[inds, :]
            overlaps = overlaps[:, inds]

            # sample done
            ################

        max_overlap_v, max_overlap_idx = torch.max(overlaps,dim=-1)

        pseudo_labels = gt_labels[max_overlap_idx]
        loss_weights = gt_weights[max_overlap_idx]
        pseudo_iou_label = max_overlap_v

        if self.test_mode:
            bg_inds = (max_overlap_v != 1)
            pseudo_labels[bg_inds,:] = 0
            pseudo_labels[bg_inds,0] = 1
            pseudo_labels = pseudo_labels * loss_weights[:, None]

            return pseudo_labels, pseudo_iou_label, loss_weights, None

        else:
            ignore_inds = max_overlap_v == 0
            pseudo_labels[ignore_inds, :] = 0
            loss_weights[ignore_inds] = 0

            bg_inds = (max_overlap_v < self.iou_th) * ~ignore_inds
            pseudo_labels[bg_inds,:] = 0
            pseudo_labels[bg_inds,0] = 1

            try:
                # 将太大的proposal标记为bg
                big_proposal = ~asy_iou_flag
                pseudo_labels[big_proposal, :] = 0
                pseudo_labels[big_proposal, 0] = 1
            except:
                pass

            # 将 pseudo_iou_label 设置为离散值
            pseudo_iou_label[pseudo_iou_label > self.full_thr] = 1
            pseudo_iou_label[pseudo_iou_label <= self.full_thr] = 0

            # # 将 pseudo_iou_label 设置为连续值
            # pseudo_iou_label = (pseudo_iou_label - self.iou_th) / (1 - self.iou_th)

            group_assign = max_overlap_idx + 1 # 将 index 从 1 开始
            group_assign[bg_inds] = -1
            group_assign[ignore_inds] = -2
            # 所以最后的范围是 [-2, -1, , 1, 2, 3...]

            # transform pseudo_labels --> N * 21
            # 将 group_assign 变为 和 初始伪标签 mat 相同的形态
            group_assign = group_assign[:,None] * pseudo_labels

            return pseudo_labels, pseudo_iou_label, loss_weights, group_assign

# # # 将 mist_layer 的 iou label 进行改进 变为矩阵的形式
# # class mist_layer_iou_DC(mist_layer):
# #     def __init__(self, portion=0.1, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85,sample=False,test_mode=False,get_diffuse_gt=False):
# #         super().__init__(portion, full_thr, iou_thr, asy_iou_th, sample, test_mode, get_diffuse_gt)
# #
# #         print("mist_layer_iou_DC")
# #
# #     @torch.no_grad()
# #     def forward(self, predict_cls, predict_det, rois, labels, iou_map=None, asy_iou_map=None, diffuse = False):
# #         if rois.ndim == 3:
# #             rois = rois.squeeze(0)
# #         rois = rois[:,1:] # remove batch_id
# #
# #         if diffuse:
# #             gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag = self.mist_label_diffuse(predict_cls, predict_det, rois, labels, iou_map, asy_iou_map)
# #         else:
# #             if predict_det!= None:
# #                 preds = predict_cls * predict_det
# #             else:
# #                 preds = predict_cls
# #
# #             gt_boxes, gt_labels, gt_weights, gt_idxs = self.mist_label(preds,rois,labels,iou_map,asy_iou_map)
# #
# #         if gt_idxs.sum() == 0:
# #             return None, None, None, None, None
# #
# #         if iou_map == None:
# #             overlaps = box_iou(rois, gt_boxes) # (proposal_num,gt_num)
# #         else:
# #             overlaps = iou_map[:, gt_idxs]
# #
# #         max_overlap_v, max_overlap_idx = torch.max(overlaps,dim=-1)
# #
# #         pseudo_labels = gt_labels[max_overlap_idx]
# #         loss_weights = gt_weights[max_overlap_idx]
# #         # pseudo_iou_label = max_overlap_v
# #         # 将 pseudo_iou_label 转为 matrix 的形式
# #         pseudo_iou_label = pseudo_labels * max_overlap_v[:, None]
# #         # 由于注册 pseudo_iou_label 的对应关系
# #         group_iou_relationship = pseudo_labels
# #
# #         if self.test_mode:
# #             bg_inds = (max_overlap_v != 1)
# #             pseudo_labels[bg_inds,:] = 0
# #             pseudo_labels[bg_inds,0] = 1
# #
# #             pseudo_iou_label = max_overlap_v
# #
# #             return pseudo_labels, pseudo_iou_label, loss_weights, None
# #
# #         else:
# #             ignore_inds = max_overlap_v == 0
# #             pseudo_labels[ignore_inds, :] = 0
# #             loss_weights[ignore_inds] = 0
# #
# #             bg_inds = (max_overlap_v < self.iou_th) * ~ignore_inds
# #             pseudo_labels[bg_inds,:] = 0
# #             pseudo_labels[bg_inds,0] = 1
# #
# #             # pseudo_iou_label[bg_inds] = 0 # bg --> iou: 0
# #             group_iou_relationship[max_overlap_v < 0.25,:] = 0
# #
# #             try:
# #                 # 将太大的proposal标记为bg
# #                 big_proposal = ~asy_iou_flag
# #                 pseudo_labels[big_proposal, :] = 0
# #                 pseudo_labels[big_proposal, 0] = 1
# #             except:
# #                 pass
# #
# #             # # 将 pseudo_iou_label 设置为离散值
# #             # pseudo_iou_label[pseudo_iou_label > self.full_thr] = 1
# #             # pseudo_iou_label[pseudo_iou_label <= self.full_thr] = 0
# #
# #             # # 将 pseudo_iou_label 设置为连续值
# #             # pseudo_iou_label = (pseudo_iou_label - self.iou_th) / (1 - self.iou_th)
# #
# #             # # 将 pseudo_iou_label 设置为离散/连续值
# #             # # 大于 full_thr 阈值的 设置为 离散值
# #             # # 介于 full_thr 和 iou_th 之间的 为连续值
# #             # pseudo_iou_label[pseudo_iou_label > self.full_thr] = 1
# #             # # [iou_th, full_thr] --> continue
# #             # # pseudo_iou_label = (pseudo_iou_label - self.iou_th) / (self.full_thr - self.iou_th)
# #             # flag = ((pseudo_iou_label <= self.full_thr) * (pseudo_iou_label >= self.iou_th)) == 1
# #             # pseudo_iou_label[flag] = 0.5
# #             # # clamp
# #             # pseudo_iou_label = torch.clamp(pseudo_iou_label,0,1)
# #
# #             # pseudo_iou_label[pseudo_iou_label >= self.full_thr] = 1
# #             # pseudo_iou_label[pseudo_iou_label < 0.5] = 0
# #             # pseudo_iou_label = (pseudo_iou_label - 0.5) / (self.full_thr - 0.5)
# #             # pseudo_iou_label = torch.clamp(pseudo_iou_label, 0, 1)
# #
# #             # pseudo_iou_label[pseudo_iou_label >= self.full_thr] = 1
# #             # pseudo_iou_label[pseudo_iou_label < 0.5] = 0
# #             # pseudo_iou_label = (pseudo_iou_label - 0.5) / (self.full_thr - 0.5)
# #             # pseudo_iou_label = torch.clamp(pseudo_iou_label, 0, 1)
# #
# #             pseudo_iou_label[pseudo_iou_label > self.full_thr] = 1
# #             pseudo_iou_label[pseudo_iou_label <= self.full_thr] = 0
# #
# #             group_assign = max_overlap_idx + 1 # 将 index 从 1 开始
# #             group_assign[bg_inds] = -1
# #             group_assign[ignore_inds] = -2
# #             # 所以最后的范围是 [-2, -1, , 1, 2, 3...]
# #
# #             # transform pseudo_labels --> N * 21
# #             # 将 group_assign 变为 和 初始伪标签 mat 相同的形态
# #             group_assign = group_assign[:,None] * pseudo_labels
# #
# #             return pseudo_labels, pseudo_iou_label, loss_weights, group_assign, group_iou_relationship
#
# # 用于统计 gt 的分布
# class mist_layer_cal(nn.Module):
#     def __init__(self, portion=0.1, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85,diffuse_mode=False):
#         super(mist_layer_cal, self).__init__()
#         self.portion = portion
#         self.full_thr = full_thr
#         self.iou_th = iou_thr
#         self.asy_iou_th = asy_iou_th
#         self.diffuse_mode = diffuse_mode
#
#         print("mist_layer--> portion:{}, full_thr: {}, iou_thr: {}, diffuse_mode: {}".format(portion,full_thr,iou_thr,diffuse_mode))
#
#     # instance_list -> [{},{}...]
#     # {} -> {score: float, mask_id: int}
#     def instance_nms(self, instance_list, iou_map):
#         instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)
#
#         selected_instances_id = []
#         while len(instance_list) > 0:
#             src_instance = instance_list.pop(0)
#             selected_instances_id.append(src_instance["mask_id"])
#
#             src_mask_id = src_instance["mask_id"]
#
#             def iou_filter(dst_instance):
#                 dst_mask_id = dst_instance["mask_id"]
#
#                 iou = iou_map[src_mask_id][dst_mask_id]
#                 if iou < self.iou_th:
#                     return True
#                 else:
#                     return False
#
#             instance_list = list(filter(iou_filter, instance_list))
#
#         return selected_instances_id
#
#     # instance_list -> [{},{}...]
#     # {} -> {score: float, mask_id: int}
#     def instance_asy_nms(self, instance_list, asy_iou_map):
#         instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)
#
#         selected_instances_id = []
#         while len(instance_list) > 0:
#             src_instance = instance_list.pop(0)
#             selected_instances_id.append(src_instance["mask_id"])
#
#             src_mask_id = src_instance["mask_id"]
#
#             def iou_filter(dst_instance):
#                 dst_mask_id = dst_instance["mask_id"]
#
#                 iou = asy_iou_map[src_mask_id][dst_mask_id]
#                 if iou < self.asy_iou_th:
#                     return True
#                 else:
#                     return False
#
#             instance_list = list(filter(iou_filter, instance_list))
#
#         return selected_instances_id
#
#     @torch.no_grad()
#     def mist_label(self, preds, rois, label, iou_map=None, asy_iou_map=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         preds = (preds if preds.shape[-1] == label.shape[-1] else preds[:, 1:]).clone()  # remove background class if present
#         keep_count = int(np.ceil(self.portion * preds.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((preds.shape[0], label.shape[-1] + 1), dtype=preds.dtype, device=preds.device)
#         gt_weights = -torch.ones((preds.shape[0],), dtype=preds.dtype, device=preds.device)
#
#         for c in klasses:
#             cls_prob_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=preds.device)
#
#             # box nms
#             else:
#                 print("iou_map == None")
#                 keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             is_higher_scoring_class = cls_prob_tmp[keep_nms_idx] > gt_weights[keep_nms_idx]
#             keep_idxs = keep_nms_idx[is_higher_scoring_class]
#             gt_labels[keep_idxs, :] = 0
#             gt_labels[keep_idxs, c + 1] = 1
#             gt_weights[keep_idxs] = cls_prob_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs
#
#     @torch.no_grad()
#     def mist_label_diffuse(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
#         predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present
#
#         preds = predict_cls * predict_det
#
#         keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
#         gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)
#
#         # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
#         asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]
#
#         for c in klasses:
#             cls_prob_tmp = predict_cls[:, c]
#             if predict_det.shape[-1] == label.shape[-1]:
#                 det_prob_tmp = predict_det[:, c]
#             elif predict_det.shape[-1] == 1:
#                 det_prob_tmp = predict_det[:, 0]
#             else:
#                 raise AssertionError
#
#             preds_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # cal iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)
#
#             # box nms
#             else:
#                 print("iou_map == None")
#                 keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             assert asy_iou_map != None
#             temp_asy_iou_map = asy_iou_map[:, keep_nms_idx]
#             temp_asy_iou_map = temp_asy_iou_map > self.asy_iou_th
#
#             flag = temp_asy_iou_map * asy_iou_flag
#             if flag.sum() != 0:
#                 col_flag = torch.sum(flag,dim=0) > 0
#                 flag = flag[:, col_flag]
#                 if self.diffuse_mode:
#                     res_det = flag * det_prob_tmp[:,None]
#                     res_idx = torch.argmax(res_det, dim=0) # 真正的 gt
#                     res_idx = torch.unique(res_idx) # diffuse 以后的 res
#                 else:
#                     res_idx = keep_nms_idx[col_flag]  # 原先的 res_idx --> 未diffuse
#
#                 is_higher_scoring_class = preds_tmp[res_idx] > gt_weights[res_idx]
#                 if is_higher_scoring_class.sum() > 0:
#                     keep_idxs = res_idx[is_higher_scoring_class]
#                     gt_labels[keep_idxs, :] = 0
#                     gt_labels[keep_idxs, c + 1] = 1
#                     gt_weights[keep_idxs] = preds_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#         # assert gt_idxs.sum() > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag
#
#     # == 0, ignore
#     # != 0 and < 0.25, bg
#     # >= 0.25, fg
#     # < 0.5, complete
#     # >= 0.5, not complete
#     @torch.no_grad()
#     def forward(self, predict_cls, predict_det, rois, labels, iou_map=None, asy_iou_map=None, diffuse = False):
#         if rois.ndim == 3:
#             rois = rois.squeeze(0)
#         rois = rois[:,1:] # remove batch_id
#
#         if diffuse:
#             gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag = self.mist_label_diffuse(predict_cls, predict_det, rois, labels, iou_map, asy_iou_map)
#         else:
#             if predict_det!= None:
#                 preds = predict_cls * predict_det
#             else:
#                 preds = predict_cls
#
#             gt_boxes, gt_labels, gt_weights, gt_idxs = self.mist_label(preds,rois,labels,iou_map,asy_iou_map)
#
#         if gt_idxs.sum() == 0:
#             return None, None, None, None
#
#         if iou_map == None:
#             overlaps = box_iou(rois, gt_boxes) # (proposal_num,gt_num)
#         else:
#             overlaps = iou_map[:, gt_idxs]
#
#         max_overlap_v, max_overlap_idx = torch.max(overlaps,dim=-1)
#
#         pseudo_labels = gt_labels[max_overlap_idx]
#         loss_weights = gt_weights[max_overlap_idx]
#         pseudo_iou_label = max_overlap_v
#
#         bg_inds = (max_overlap_v != 1)
#         pseudo_labels[bg_inds,:] = 0
#         pseudo_labels[bg_inds,0] = 1
#
#         return pseudo_labels, pseudo_iou_label, loss_weights, None

# class mist_layer_wsod2(mist_layer):
#     def __init__(self, portion=0.1, scale=0.5, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85,sample=False,test_mode=False,get_diffuse_gt=False):
#         super().__init__(portion, full_thr, iou_thr, asy_iou_th, sample ,test_mode, get_diffuse_gt)
#
#         self.scale = scale
#
#         print("mist_layer_wsod2--> scale: {}".format(scale))
#
#     @torch.no_grad()
#     def mist_label_diffuse(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
#         predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present
#
#         preds = predict_cls * predict_det
#
#         # keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
#         gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)
#
#         # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
#         asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]
#
#         for c in klasses:
#             cls_prob_tmp = predict_cls[:, c]
#             if predict_det.shape[-1] == label.shape[-1]:
#                 det_prob_tmp = predict_det[:, c]
#             elif predict_det.shape[-1] == 1:
#                 det_prob_tmp = predict_det[:, 0]
#             else:
#                 raise AssertionError
#
#             preds_tmp = preds[:, c]
#
#             keep_count = min(int(torch.sum(cls_prob_tmp >= (torch.max(cls_prob_tmp) * self.scale))),
#                              int(np.ceil(self.portion * predict_cls.shape[0])))
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # cal iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)
#
#             # box nms
#             else:
#                 print("iou_map == None")
#                 keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             assert asy_iou_map != None
#             temp_asy_iou_map = asy_iou_map[:, keep_nms_idx]
#             temp_asy_iou_map = temp_asy_iou_map > self.asy_iou_th
#
#             flag = temp_asy_iou_map * asy_iou_flag
#             if flag.sum() != 0:
#                 flag = flag[:, torch.sum(flag,dim=0) > 0]
#                 res_det = flag * det_prob_tmp[:,None]
#                 res_idx = torch.argmax(res_det, dim=0) # 真正的 gt
#                 res_idx = torch.unique(res_idx)
#
#                 is_higher_scoring_class = preds_tmp[res_idx] > gt_weights[res_idx]
#                 if is_higher_scoring_class.sum() > 0:
#                     keep_idxs = res_idx[is_higher_scoring_class]
#                     gt_labels[keep_idxs, :] = 0
#                     gt_labels[keep_idxs, c + 1] = 1
#                     gt_weights[keep_idxs] = preds_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#         # assert gt_idxs.sum() > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag

class mist_layer_visual(nn.Module):
    def __init__(self, portion=0.1, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85, dir_path=None):
        super(mist_layer_visual, self).__init__()
        self.portion = portion
        self.full_thr = full_thr
        self.iou_th = iou_thr
        self.asy_iou_th = asy_iou_th
        self.dir_path = dir_path

        print("mist_layer_visual--> full_thr: {}, iou_thr: {}".format(full_thr,iou_thr))
        print("dir path: {}".format(dir_path))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # instance_list -> [{},{}...]
    # {} -> {score: float, mask_id: int}
    def instance_nms(self, instance_list, iou_map):
        instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)

        selected_instances_id = []
        while len(instance_list) > 0:
            src_instance = instance_list.pop(0)
            selected_instances_id.append(src_instance["mask_id"])

            src_mask_id = src_instance["mask_id"]

            def iou_filter(dst_instance):
                dst_mask_id = dst_instance["mask_id"]

                iou = iou_map[src_mask_id][dst_mask_id]
                if iou < self.iou_th:
                    return True
                else:
                    return False

            instance_list = list(filter(iou_filter, instance_list))

        return selected_instances_id

    # instance_list -> [{},{}...]
    # {} -> {score: float, mask_id: int}
    def instance_asy_nms(self, instance_list, asy_iou_map):
        instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)

        selected_instances_id = []
        while len(instance_list) > 0:
            src_instance = instance_list.pop(0)
            selected_instances_id.append(src_instance["mask_id"])

            src_mask_id = src_instance["mask_id"]

            def iou_filter(dst_instance):
                dst_mask_id = dst_instance["mask_id"]

                iou = asy_iou_map[src_mask_id][dst_mask_id]
                if iou < self.asy_iou_th:
                    return True
                else:
                    return False

            instance_list = list(filter(iou_filter, instance_list))

        return selected_instances_id

    @torch.no_grad()
    def mist_label(self, preds, rois, label, iou_map=None, asy_iou_map=None):
        if label.dim() != 1:
            label = label.squeeze()

        assert label.dim() == 1
        assert label.shape[-1] == 20 or label.shape[-1] == 80

        # bg remove
        preds = (preds if preds.shape[-1] == label.shape[-1] else preds[:, 1:]).clone()  # remove background class if present
        keep_count = int(np.ceil(self.portion * preds.shape[0]))
        klasses = label.nonzero(as_tuple=True)[0]
        # one hot label
        gt_labels = torch.zeros((preds.shape[0], label.shape[-1] + 1), dtype=preds.dtype, device=preds.device)
        gt_weights = -torch.ones((preds.shape[0],), dtype=preds.dtype, device=preds.device)

        for c in klasses:
            cls_prob_tmp = preds[:, c]

            keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals

            keep_rois = rois[keep_sort_idx]
            keep_cls_prob = cls_prob_tmp[keep_sort_idx]

            # iou nms
            if iou_map != None:
                temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]

                instance_list = []
                for i, prob in enumerate(keep_cls_prob):
                    instance = dict()

                    instance["score"] = prob
                    instance["mask_id"] = i
                    instance_list.append(instance)

                keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
                keep_nms_idx = torch.tensor(keep_nms_idx,device=preds.device)

            # box nms
            else:
                print("iou_map == None")
                keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)

            keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index

            is_higher_scoring_class = cls_prob_tmp[keep_nms_idx] > gt_weights[keep_nms_idx]
            keep_idxs = keep_nms_idx[is_higher_scoring_class]
            gt_labels[keep_idxs, :] = 0
            gt_labels[keep_idxs, c + 1] = 1
            gt_weights[keep_idxs] = cls_prob_tmp[keep_idxs]

        gt_idxs = torch.sum(gt_labels, dim=-1) > 0

        gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]

        return gt_boxes, gt_labels, gt_weights, gt_idxs

    @torch.no_grad()
    def mist_label_diffuse(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None):
        if label.dim() != 1:
            label = label.squeeze()

        assert label.dim() == 1
        assert label.shape[-1] == 20 or label.shape[-1] == 80

        # bg remove
        predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
        predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present

        preds = predict_cls * predict_det

        keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))
        klasses = label.nonzero(as_tuple=True)[0]
        # one hot label
        gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
        gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)

        # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
        asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]

        org_gt_idxs = []
        diffuse_gt_idxs = []
        org_cls_score = []
        diffuse_cls_score = []
        org_iou_score = []
        diffuse_iou_score = []

        class_cat = []

        for c in klasses:
            cls_prob_tmp = predict_cls[:, c]
            if predict_det.shape[-1] == label.shape[-1]:
                det_prob_tmp = predict_det[:, c]
            elif predict_det.shape[-1] == 1:
                det_prob_tmp = predict_det[:, 0]
            else:
                raise AssertionError

            preds_tmp = preds[:, c]

            keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals

            keep_rois = rois[keep_sort_idx]
            keep_cls_prob = cls_prob_tmp[keep_sort_idx]

            # cal iou nms
            if iou_map != None:
                temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]

                instance_list = []
                for i, prob in enumerate(keep_cls_prob):
                    instance = dict()

                    instance["score"] = prob
                    instance["mask_id"] = i
                    instance_list.append(instance)

                keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
                keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)

            # box nms
            else:
                print("iou_map == None")
                keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)

            keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index

            assert asy_iou_map != None
            temp_asy_iou_map = asy_iou_map[:, keep_nms_idx]
            temp_asy_iou_map = temp_asy_iou_map > self.asy_iou_th

            flag = temp_asy_iou_map * asy_iou_flag
            if flag.sum() != 0:
                f = torch.sum(flag,dim=0) > 0
                flag = flag[:, f]
                print(keep_nms_idx[f])
                org_gt_idxs.append(keep_nms_idx[f])
                org_cls_score.append(cls_prob_tmp[keep_nms_idx[f]])
                org_iou_score.append(det_prob_tmp[keep_nms_idx[f]])
                res_det = flag * det_prob_tmp[:,None]
                res_idx = torch.argmax(res_det, dim=0) # 真正的 gt
                diffuse_gt_idxs.append(res_idx)
                diffuse_cls_score.append(cls_prob_tmp[res_idx])
                diffuse_iou_score.append(det_prob_tmp[res_idx])

                class_cat.append(torch.ones_like(det_prob_tmp[res_idx])*c)
                res_idx = torch.unique(res_idx)

                is_higher_scoring_class = preds_tmp[res_idx] > gt_weights[res_idx]
                if is_higher_scoring_class.sum() > 0:
                    keep_idxs = res_idx[is_higher_scoring_class]
                    gt_labels[keep_idxs, :] = 0
                    gt_labels[keep_idxs, c + 1] = 1
                    gt_weights[keep_idxs] = preds_tmp[keep_idxs]

        gt_idxs = torch.sum(gt_labels, dim=-1) > 0
        # assert gt_idxs.sum() > 0

        gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]

        try:
            org_gt_idxs = torch.concat(org_gt_idxs)
            diffuse_gt_idxs = torch.concat(diffuse_gt_idxs)
            org_cls_score = torch.concat(org_cls_score)
            org_iou_score = torch.concat(org_iou_score)
            diffuse_cls_score = torch.concat(diffuse_cls_score)
            diffuse_iou_score = torch.concat(diffuse_iou_score)
            class_cat = torch.concat(class_cat)
        except:
            pass

        return gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag, \
            org_gt_idxs,diffuse_gt_idxs, org_cls_score,org_iou_score, \
            diffuse_cls_score,diffuse_iou_score,class_cat


    # == 0, ignore
    # != 0 and < 0.25, bg
    # >= 0.25, fg
    # < 0.5, complete
    # >= 0.5, not complete
    @torch.no_grad()
    def forward(self, predict_cls, predict_det, rois, labels, iou_map, asy_iou_map,image_name, diffuse = False):
        if rois.ndim == 3:
            rois = rois.squeeze(0)
        rois = rois[:,1:] # remove batch_id

        scale = 0.5
        # VOC
        if os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/JPEGImages",image_name)):
            cls_name_map,cls_id_map = id_2_clsname("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/annotations/voc_2012_val.json")

            image = Image.open(os.path.join("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/JPEGImages",image_name)).convert('RGB')
            # img
            s = image_name.replace(".jpg","")

            # proposal
            COB_proposals = loadmat(os.path.join("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/COB_SBD_trainaug", s + '.mat'))['maskmat'][:, 0]

            mask_proposals = [np.array(p) for p in COB_proposals]
            masks = np.array(mask_proposals)

        # COCO
        elif os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/train2017",image_name)):
            cls_name_map,cls_id_map = id_2_clsname("/home/lzc/WSIS-Benchmark/dataset/coco2017/annotations/instances_val2017.json")

            image = Image.open(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/train2017",image_name)).convert('RGB')
            # proposal
            s = image_name.replace(".jpg","")

            file_name = 'COCO_train2014_' + s + '.mat'
            if not os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/COB-COCO", file_name)):
                file_name = 'COCO_val2014_' + s + '.mat'
            if not os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/COB-COCO", file_name)):
                file_name = s + '.mat'
            COB_proposals = loadmat(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/COB-COCO", file_name),
                                             verify_compressed_data_integrity=False)['maskmat']

            mask_proposals = [np.array(p) for p in COB_proposals]
            masks = np.array(mask_proposals)
        else:
            raise AssertionError

        if diffuse:
            gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag, \
            org_gt_idxs,diffuse_gt_idxs, org_cls_score,org_iou_score, \
            diffuse_cls_score,diffuse_iou_score,class_cat = self.mist_label_diffuse(predict_cls, predict_det, rois, labels, iou_map, asy_iou_map)

            org_color = np.array([[0.000, 0.000, 1.000]])
            diffuse_color = np.array([[1.000, 1.000, 0.000]])

            # cls_label = (torch.nonzero(gt_labels.flatten()) % gt_labels.shape[-1]).flatten() - 1

            for cls_idx in torch.unique(class_cat):
                cls_idx = int(cls_idx)
                temp_diffuse_gt_idxs = diffuse_gt_idxs[class_cat == cls_idx]
                temp_org_gt_idxs = org_gt_idxs[class_cat == cls_idx]
                temp_org_cls_score = org_cls_score[class_cat == cls_idx]
                temp_org_iou_score = org_iou_score[class_cat == cls_idx]
                temp_diffuse_cls_score = diffuse_cls_score[class_cat == cls_idx]
                temp_diffuse_iou_score = diffuse_iou_score[class_cat == cls_idx]

                result_list = []

                for kkk,(diffuse_idx,org_idx) in enumerate(zip(temp_diffuse_gt_idxs,temp_org_gt_idxs)):
                    proposal = masks[diffuse_idx]

                    ind_xy = np.nonzero(proposal)
                    xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1

                    label_list = []
                    label_list.append(cls_name_map[cls_idx] + "--> cls:%.2f, iou:%.2f" % (
                    (predict_cls[diffuse_idx,cls_idx + 1].item(),predict_det[diffuse_idx,cls_idx + 1].item())))
                    assert predict_cls[diffuse_idx,cls_idx + 1].item() == temp_diffuse_cls_score[kkk] and \
                           predict_det[diffuse_idx,cls_idx + 1].item() == temp_diffuse_iou_score[kkk]

                    diifuse_res = imshow_det_bboxes(np.array(image), np.array([[xmin, ymin, xmax, ymax]]),
                                               labels=np.array([0]), segms=proposal[None, ...],
                                               class_names=label_list, show=False,
                                               bbox_color=diffuse_color, mask_color=255 * diffuse_color,
                                               font_size=18, thickness=1)

                    if org_idx == diffuse_idx:
                        label_list = []
                        label_list.append("off-diffuse" + "--> cls:%.2f, iou:%.2f" % (
                            predict_cls[org_idx, cls_idx + 1].item(),
                            predict_det[org_idx, cls_idx + 1].item()))

                    else:
                        label_list = []
                        label_list.append("on-diffuse" + "--> cls:%.2f, iou:%.2f" % (
                        predict_cls[org_idx, cls_idx + 1].item(),
                        predict_det[org_idx, cls_idx + 1].item()))
                        assert predict_cls[org_idx, cls_idx + 1].item() == temp_org_cls_score[kkk] and \
                               predict_det[org_idx, cls_idx + 1].item() == temp_org_iou_score[kkk]

                    proposal = masks[org_idx]

                    ind_xy = np.nonzero(proposal)
                    xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1

                    org_diifuse_res = imshow_det_bboxes(np.array(image), np.array([[xmin, ymin, xmax, ymax]]),
                                                    labels=np.array([0]), segms=proposal[None, ...],
                                                    class_names=label_list, show=False,
                                                    bbox_color=org_color, mask_color=255 * org_color,
                                                    font_size=18, thickness=1)

                    result_list.append(np.concatenate((org_diifuse_res,diifuse_res),axis=1))

                visual_result = np.concatenate(result_list,axis=0)
                plt.imshow(visual_result)
                plt.axis("off")
                plt.savefig(os.path.join(self.dir_path, image_name.replace(".jpg",cls_name_map[cls_idx]+".jpg")), dpi=100*len(result_list), bbox_inches='tight')

        else:
            if predict_det!= None:
                preds = predict_cls * predict_det
            else:
                preds = predict_cls

            gt_boxes, gt_labels, gt_weights, gt_idxs = self.mist_label(preds,rois,labels,iou_map,asy_iou_map)
            gt_boxes = gt_boxes.detach()
            gt_labels = gt_labels.detach()
            gt_weights = gt_weights.detach()
            gt_idxs = gt_idxs.detach()

            cls_label = (torch.nonzero(gt_labels.flatten()) % gt_labels.shape[-1]).flatten() - 1
            gt_idxs = torch.nonzero(gt_idxs).flatten()

            order = gt_weights.cpu().numpy().argsort()[::-1] # down
            cls_label = cls_label.cpu().numpy()[order]
            gt_weights = gt_weights.cpu().numpy()[order]
            gt_idxs = gt_idxs.cpu().numpy()[order]

            print(cls_label)
            for cls_idx in np.unique(cls_label):
                idxs = gt_idxs[cls_label == cls_idx]
                weights = gt_weights[cls_label == cls_idx]
                result_list = []

                for iii,idx in enumerate(idxs):
                    label_list = []

                    proposal = masks[idx]
                    ind_xy = np.nonzero(proposal)
                    xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1

                    org_color = np.array([[0.000, 0.000, 1.000]])
                    diffuse_color = np.array([[1.000, 1.000, 0.000]])

                    # proposal = proposal[...,None].repeat(3, axis=-1) * diffuse_color[None, None,:]
                    # heatmap = np.uint8(255 * proposal)
                    #
                    # result = np.uint8(heatmap * scale + np.array(image) * (1 - scale))
                    # result = cv2.rectangle(result, (xmin, ymin), (xmax, ymax), diffuse_color*255)
                    # result = cv2.putText(result, "cls*iou:%.2f"%weights[iii], (xmin+1, ymin-1), cv2.FONT_HERSHEY_COMPLEX, 0.5, diffuse_color*255)

                    label_list.append(cls_name_map[cls_idx]+":%.2f"%weights[iii])
                    label_list_np = np.array([0])

                    result = imshow_det_bboxes(np.array(image),np.array([[xmin,ymin,xmax,ymax]]),
                                               labels=label_list_np,segms=proposal[None,...],
                                               class_names=label_list,show=False,
                                               bbox_color=diffuse_color,mask_color=255 * diffuse_color,
                                               font_size=18,thickness=1)

                    result_list.append(np.uint8(result))

                visual_result = np.concatenate(result_list,axis=0)
                plt.close()
                plt.imshow(visual_result)
                plt.axis("off")
                plt.savefig(os.path.join(self.dir_path, image_name.replace(".jpg",(cls_name_map[cls_idx]+".jpg"))), dpi=100*len(result_list), bbox_inches='tight')

# class mist_layer_visual_bu(nn.Module):
#     def __init__(self, portion=0.1, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85, dir_path=None):
#         super(mist_layer_visual_bu, self).__init__()
#         self.portion = portion
#         self.full_thr = full_thr
#         self.iou_th = iou_thr
#         self.asy_iou_th = asy_iou_th
#         self.dir_path = dir_path
#
#         print("mist_layer_visual_bu--> full_thr: {}, iou_thr: {}".format(full_thr, iou_thr))
#         print("dir path: {}".format(dir_path))
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path)
#
#     # instance_list -> [{},{}...]
#     # {} -> {score: float, mask_id: int}
#     def instance_nms(self, instance_list, iou_map):
#         instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)
#
#         selected_instances_id = []
#         while len(instance_list) > 0:
#             src_instance = instance_list.pop(0)
#             selected_instances_id.append(src_instance["mask_id"])
#
#             src_mask_id = src_instance["mask_id"]
#
#             def iou_filter(dst_instance):
#                 dst_mask_id = dst_instance["mask_id"]
#
#                 iou = iou_map[src_mask_id][dst_mask_id]
#                 if iou < self.iou_th:
#                     return True
#                 else:
#                     return False
#
#             instance_list = list(filter(iou_filter, instance_list))
#
#         return selected_instances_id
#
#     # instance_list -> [{},{}...]
#     # {} -> {score: float, mask_id: int}
#     def instance_asy_nms(self, instance_list, asy_iou_map):
#         instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)
#
#         selected_instances_id = []
#         while len(instance_list) > 0:
#             src_instance = instance_list.pop(0)
#             selected_instances_id.append(src_instance["mask_id"])
#
#             src_mask_id = src_instance["mask_id"]
#
#             def iou_filter(dst_instance):
#                 dst_mask_id = dst_instance["mask_id"]
#
#                 iou = asy_iou_map[src_mask_id][dst_mask_id]
#                 if iou < self.asy_iou_th:
#                     return True
#                 else:
#                     return False
#
#             instance_list = list(filter(iou_filter, instance_list))
#
#         return selected_instances_id
#
#     @torch.no_grad()
#     def mist_label(self, preds, rois, label, iou_map=None, asy_iou_map=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         preds = (preds if preds.shape[-1] == label.shape[-1] else preds[:, 1:]).clone()  # remove background class if present
#         keep_count = int(np.ceil(self.portion * preds.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((preds.shape[0], label.shape[-1] + 1), dtype=preds.dtype, device=preds.device)
#         gt_weights = -torch.ones((preds.shape[0],), dtype=preds.dtype, device=preds.device)
#
#         for c in klasses:
#             cls_prob_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=preds.device)
#
#             # box nms
#             else:
#                 print("iou_map == None")
#                 keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             is_higher_scoring_class = cls_prob_tmp[keep_nms_idx] > gt_weights[keep_nms_idx]
#             keep_idxs = keep_nms_idx[is_higher_scoring_class]
#             gt_labels[keep_idxs, :] = 0
#             gt_labels[keep_idxs, c + 1] = 1
#             gt_weights[keep_idxs] = cls_prob_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs
#
#     @torch.no_grad()
#     def mist_label_diffuse(self, predict_cls, predict_det, rois, label, iou_map, asy_iou_map, bu_cues,masks,image,cls_name_map,image_name):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
#         predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present
#
#         preds = predict_cls * predict_det
#
#         keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
#         gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)
#
#         # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
#         asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]
#
#         for c in klasses:
#             cls_prob_tmp = predict_cls[:, c]
#             if predict_det.shape[-1] == label.shape[-1]:
#                 det_prob_tmp = predict_det[:, c]
#             elif predict_det.shape[-1] == 1:
#                 det_prob_tmp = predict_det[:, 0]
#             else:
#                 raise AssertionError
#
#             preds_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # cal iou nms
#             temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#             instance_list = []
#             for i, prob in enumerate(keep_cls_prob):
#                 instance = dict()
#
#                 instance["score"] = prob
#                 instance["mask_id"] = i
#                 instance_list.append(instance)
#
#             keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#             keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             assert asy_iou_map != None
#             temp_asy_iou_map = asy_iou_map[:, keep_nms_idx]
#             temp_asy_iou_map = temp_asy_iou_map > self.asy_iou_th
#
#             flag = temp_asy_iou_map * asy_iou_flag
#             org_det = det_prob_tmp[keep_nms_idx]
#
#             if flag.sum() != 0:
#                 col_idxs = torch.sum(flag,dim=0) > 0
#                 flag = flag[:, col_idxs]
#                 org_det = org_det[col_idxs][None,:] # 1, N
#                 org_idxs = keep_nms_idx[col_idxs]
#
#                 res_det = flag * det_prob_tmp[:,None]
#
#                 diffuse_idxs = res_det > org_det
#                 diffuse_flag = torch.sum(diffuse_idxs,dim=0,keepdim=True) > 0
#                 score_map = (diffuse_flag * diffuse_idxs * bu_cues[:, None] + (~diffuse_flag) * res_det)
#                 res_score, res_idxs = torch.topk(
#                                     score_map,
#                                     k=3,dim=0) # 真正的 gt
#
#                 # score_map = (res_det * bu_cues[:, None] * cls_prob_tmp[:,None])
#                 # res_score, res_idxs = torch.topk(
#                 #     score_map,
#                 #     k=3, dim=0)  # 真正的 gt
#
#                 org_res_score, org_res_idxs = torch.topk(res_det,k=1, dim=0)  # 真正的 gt
#
#                 org_color = np.array([[0.000, 0.000, 1.000]])
#                 diffuse_color = np.array([[1.000, 1.000, 0.000]])
#
#                 pic_board = []
#                 for idx, org_idx in enumerate(org_idxs):
#                     proposal = masks[org_idx]
#
#                     ind_xy = np.nonzero(proposal)
#                     xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
#
#                     label_list = []
#                     label_list.append("c:%.2f, i:%.2f, u:%.2f" % (
#                         (cls_prob_tmp[org_idx].item(), det_prob_tmp[org_idx].item(),bu_cues[org_idx])))
#                     org_res = imshow_det_bboxes(np.array(image), np.array([[xmin, ymin, xmax, ymax]]),
#                                                     labels=np.array([0]), segms=proposal[None, ...],
#                                                     class_names=label_list, show=False,
#                                                     bbox_color=org_color, mask_color=255 * org_color,
#                                                     font_size=18, thickness=1)
#
#                     proposal = masks[org_res_idxs[0,idx]]
#
#                     ind_xy = np.nonzero(proposal)
#                     xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
#
#                     label_list = []
#                     label_list.append("c:%.2f, i:%.2f, u:%.2f" % (
#                         (cls_prob_tmp[org_res_idxs[0,idx]].item(), det_prob_tmp[org_res_idxs[0,idx]].item(),bu_cues[org_res_idxs[0,idx]])))
#                     org_diffuse_res = imshow_det_bboxes(np.array(image), np.array([[xmin, ymin, xmax, ymax]]),
#                                                 labels=np.array([0]), segms=proposal[None, ...],
#                                                 class_names=label_list, show=False,
#                                                 bbox_color=diffuse_color, mask_color=255 * diffuse_color,
#                                                 font_size=18, thickness=1)
#
#                     col_board = []
#                     col_board.append(org_res)
#                     col_board.append(org_diffuse_res)
#                     for res_idx,score in zip(res_idxs[:,idx],res_score[:,idx]):
#                         proposal = masks[res_idx]
#
#                         ind_xy = np.nonzero(proposal)
#                         xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[
#                             0].max() + 1
#
#                         label_list = []
#                         label_list.append("c:%.2f, i:%.2f, u:%.2f" % (
#                             (cls_prob_tmp[res_idx].item(), det_prob_tmp[res_idx].item(),bu_cues[res_idx])))
#                         diffuse_res = imshow_det_bboxes(np.array(image), np.array([[xmin, ymin, xmax, ymax]]),
#                                                     labels=np.array([0]), segms=proposal[None, ...],
#                                                     class_names=label_list, show=False,
#                                                     bbox_color=diffuse_color, mask_color=255 * diffuse_color,
#                                                     font_size=18, thickness=1)
#                         col_board.append(diffuse_res)
#
#                     pic_board.append(np.concatenate(col_board,axis=1))
#
#                 visual_result = np.concatenate(pic_board,axis=0)
#
#                 plt.imshow(visual_result)
#                 plt.axis("off")
#                 plt.savefig(os.path.join(self.dir_path, image_name.replace(".jpg",cls_name_map[c]+".jpg")), dpi=int(50*(idx+1)), bbox_inches='tight')
#
#
#     # == 0, ignore
#     # != 0 and < 0.25, bg
#     # >= 0.25, fg
#     # < 0.5, complete
#     # >= 0.5, not complete
#     @torch.no_grad()
#     def forward(self, predict_cls, predict_det, rois, labels, iou_map, asy_iou_map,bu_cues, image_name, diffuse=False):
#         if rois.ndim == 3:
#             rois = rois.squeeze(0)
#         rois = rois[:, 1:]  # remove batch_id
#
#         if os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/JPEGImages",image_name)):
#             cls_name_map,cls_id_map = id_2_clsname("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/annotations/voc_2012_val.json")
#
#             image = Image.open(os.path.join("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/JPEGImages",image_name)).convert('RGB')
#             # img
#             s = image_name.replace(".jpg","")
#
#             # proposal
#             COB_proposals = loadmat(os.path.join("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/COB_SBD_trainaug", s + '.mat'))['maskmat'][:, 0]
#
#             mask_proposals = [np.array(p) for p in COB_proposals]
#             masks = np.array(mask_proposals)
#
#         # COCO
#         elif os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/train2017",image_name)):
#             cls_name_map,cls_id_map = id_2_clsname("/home/lzc/WSIS-Benchmark/dataset/coco2017/annotations/instances_val2017.json")
#
#             image = Image.open(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/train2017",image_name)).convert('RGB')
#             # proposal
#             s = image_name.replace(".jpg","")
#
#             file_name = 'COCO_train2014_' + s + '.mat'
#             if not os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/COB-COCO", file_name)):
#                 file_name = 'COCO_val2014_' + s + '.mat'
#             if not os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/COB-COCO", file_name)):
#                 file_name = s + '.mat'
#             COB_proposals = loadmat(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/COB-COCO", file_name),
#                                              verify_compressed_data_integrity=False)['maskmat']
#
#             mask_proposals = [np.array(p) for p in COB_proposals]
#             masks = np.array(mask_proposals)
#         else:
#             raise AssertionError
#         if diffuse:
#             self.mist_label_diffuse(predict_cls, predict_det,
#                                      rois, labels, iou_map,
#                                      asy_iou_map,bu_cues,masks,image,cls_name_map,image_name)
#         else:
#             raise AssertionError

# # 将 proposal 叠起来，可视化 heatmap
# class mist_layer_visual_heatmap(nn.Module):
#     def __init__(self, portion=0.1, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85, dir_path=None,PAMR_model=None):
#         super(mist_layer_visual_heatmap, self).__init__()
#         self.portion = portion
#         self.full_thr = full_thr
#         self.iou_th = iou_thr
#         self.asy_iou_th = asy_iou_th
#         self.dir_path = dir_path
#
#         print("mist_layer_visual_heatmap--> full_thr: {}, iou_thr: {}".format(full_thr, iou_thr))
#         print("dir path: {}".format(dir_path))
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path)
#
#         self.PAMR_model = PAMR_model
#
#     def pseudo_gtmask(self, mask, cutoff_top=0.6, cutoff_low=0.2, eps=1e-8):
#         """Convert continuous mask into binary mask"""
#         bs,c,h,w = mask.size()
#         mask = mask.view(bs,c,-1)
#
#         # for each class extract the max confidence
#         mask_max, _ = mask.max(-1, keepdim=True)
#         mask_max[:, :1] *= 0.7
#         mask_max[:, 1:] *= cutoff_top
#         #mask_max *= cutoff_top
#
#         # if the top score is too low, ignore it
#         # lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
#         # mask_max = mask_max.max(lowest)
#
#         pseudo_gt = (mask > mask_max).type_as(mask)
#
#         # remove ambiguous pixels
#         ambiguous = (pseudo_gt.sum(1, keepdim=True) > 1).type_as(mask)
#         pseudo_gt = (1 - ambiguous) * pseudo_gt
#
#         return pseudo_gt.view(bs,c,h,w)
#
#     # instance_list -> [{},{}...]
#     # {} -> {score: float, mask_id: int}
#     def instance_nms(self, instance_list, iou_map):
#         instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)
#
#         selected_instances_id = []
#         group_instances_id = []
#         while len(instance_list) > 0:
#             src_instance = instance_list.pop(0)
#             selected_instances_id.append(src_instance["mask_id"])
#
#             src_mask_id = src_instance["mask_id"]
#
#             keep = []
#             drop = []
#             drop.append(src_mask_id)
#
#             for dst_instance in instance_list:
#                 dst_mask_id = dst_instance["mask_id"]
#
#                 iou = iou_map[src_mask_id][dst_mask_id]
#                 if iou < self.iou_th:
#                     keep.append(dst_instance)
#
#                 else:
#                     drop.append(dst_mask_id)
#
#             instance_list = keep
#             group_instances_id.append(drop)
#
#         return selected_instances_id,group_instances_id
#
#     @torch.no_grad()
#     def mist_label(self, preds, rois, label, iou_map=None, asy_iou_map=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         preds = (preds if preds.shape[-1] == label.shape[-1] else preds[:, 1:]).clone()  # remove background class if present
#         keep_count = int(np.ceil(self.portion * preds.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((preds.shape[0], label.shape[-1] + 1), dtype=preds.dtype, device=preds.device)
#         gt_weights = -torch.ones((preds.shape[0],), dtype=preds.dtype, device=preds.device)
#
#         for c in klasses:
#             cls_prob_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=preds.device)
#
#             # box nms
#             else:
#                 print("iou_map == None")
#                 keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             is_higher_scoring_class = cls_prob_tmp[keep_nms_idx] > gt_weights[keep_nms_idx]
#             keep_idxs = keep_nms_idx[is_higher_scoring_class]
#             gt_labels[keep_idxs, :] = 0
#             gt_labels[keep_idxs, c + 1] = 1
#             gt_weights[keep_idxs] = cls_prob_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs
#
#     @torch.no_grad()
#     def mist_label_diffuse(self, predict_cls, predict_det, rois, label, iou_map, asy_iou_map, bu_cues,masks,image,cls_name_map,image_name):
#         if label.dim() != 1:
#             label = label.squeeze()
#         self.down_scale = 1
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#         image = torch.tensor(np.array(image),dtype=torch.float32,device=predict_cls.device)
#
#         image = torch.nn.functional.interpolate(image.permute(2, 0, 1)[None, ...],
#                                                 scale_factor=(self.down_scale, self.down_scale),
#                                                 mode="bilinear", align_corners=True, recompute_scale_factor=True)
#
#
#         masks = torch.tensor(masks, dtype=torch.float32, device=predict_cls.device)
#         masks = torch.nn.functional.interpolate(masks[:, None], scale_factor=(self.down_scale, self.down_scale),
#                                                     mode="bilinear", align_corners=True, recompute_scale_factor=True)
#
#         bg_prob = predict_cls[:, 0][:,None]
#
#         # bg remove
#         predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
#         predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present
#
#         preds = predict_cls * predict_det
#
#         # keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))
#
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
#         gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)
#
#         # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
#         asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]
#
#         fg_prob = predict_cls[:,klasses]
#         cam = torch.concat((bg_prob,fg_prob),dim=1) # 200,sum(labels)+1
#
#         start = time.time()
#
#         # score_map = (proposals * cam[...,None,None]).sum(0) # klasses.size() + 1
#         score_map = (masks * cam[..., None, None]).sum(0) / (1e-5 + torch.tensor(masks, device=cam.device).sum(0))  # klasses.size() + 1
#         b,h,w = score_map.shape
#         score_map = torchvision.transforms.functional.gaussian_blur(score_map, 11)
#
#         # score_map = score_map.view(b,-1)
#         # min_score = torch.min(score_map,dim=-1,keepdim=True)[0]
#         # max_score = torch.max(score_map,dim=-1,keepdim=True)[0]
#         #
#         # score_map = ((score_map - min_score) / (max_score - min_score + 1e-5)).view(b,h,w)
#         refine_score_map = self.PAMR_model(image, score_map.view(1,b,h,w))
#         # upsample
#         score_map = torch.nn.functional.interpolate(score_map.view(1,b,h,w),scale_factor=1/self.down_scale,mode="bilinear", align_corners=True,recompute_scale_factor=True)
#         up_image = torch.nn.functional.interpolate(image,scale_factor=1/self.down_scale,mode="bilinear", align_corners=True,recompute_scale_factor=True)
#         refine_score_map = torch.nn.functional.interpolate(refine_score_map,scale_factor=1/self.down_scale,mode="bilinear", align_corners=True,recompute_scale_factor=True)
#         pseudo_gt = self.pseudo_gtmask(refine_score_map)[0]
#
#         _,b,h,w = score_map.shape
#         score_map = (score_map.view(b,-1) / (torch.max(score_map.view(b,-1),dim=-1,keepdim=True)[0] + 1e-5)).view(b,h,w)
#         # print(score_map.shape)
#         # print(refine_score_map.shape)
#         refine_score_map = (refine_score_map.view(b, -1) / (torch.max(refine_score_map.view(b, -1), dim=-1, keepdim=True)[0] + 1e-5)).view(b,h,w)
#         end = time.time()
#         print(end - start, 's')
#
#         np_image = up_image[0].permute(1,2,0).cpu().numpy()
#
#         for c in klasses:
#             cls_prob_tmp = predict_cls[:, c]
#             if predict_det.shape[-1] == label.shape[-1]:
#                 det_prob_tmp = predict_det[:, c]
#             elif predict_det.shape[-1] == 1:
#                 det_prob_tmp = predict_det[:, 0]
#             else:
#                 raise AssertionError
#
#             preds_tmp = preds[:, c]
#
#             keep_count = min(int(torch.sum(cls_prob_tmp >= (torch.max(cls_prob_tmp) * 0.3))),
#                              int(np.ceil(self.portion * predict_cls.shape[0])))
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # cal iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx,group_instances_ids = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)
#
#             # box nms
#             else:
#                 raise AssertionError
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             for iik, group_instances_id in enumerate(group_instances_ids):
#                 print(group_instances_id)
#                 real_group_id = keep_sort_idx[torch.tensor(group_instances_id, device=predict_cls.device)]
#                 cam = torch.sum(masks[real_group_id] *
#                                 cls_prob_tmp[real_group_id, None, None, None], dim=0) \
#                                 # / (1e-5 + masks[real_group_id].sum(0))
#
#                 cam /= torch.max(cam)
#
#                 # cam -= torch.tensor(masks[real_group_id].sum(0)==0,dtype=torch.float)
#                 cam = torchvision.transforms.functional.gaussian_blur(cam,11)
#
#                 # refine_cam = self.PAMR_model(image, cam[None,...]) + 1
#                 #
#                 # refine_cam /= (1e-5 + torch.max(refine_cam))
#                 # cam = cam + 1
#                 # cam /= (1e-5 + torch.max(cam))
#                 #
#                 refine_cam = self.PAMR_model(image, cam[None,...])
#                 # refine_cam = torch.nn.functional.relu(refine_cam-cam[None,...])
#                 refine_cam /= (1e-5 + torch.max(refine_cam))
#                 # refine_cam = torch.nn.functional.interpolate(refine_cam, scale_factor=1 / self.down_scale,
#                 #                                                    mode="bilinear", align_corners=True,
#                 #                                                    recompute_scale_factor=True)
#
#                 heatmap = cv2.applyColorMap(np.uint8(255 * cam[0].cpu().numpy()), cv2.COLORMAP_JET)
#                 heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#                 result = heatmap * 0.7 + image[0].permute(1, 2, 0).cpu().numpy() * (1 - 0.7)
#                 result_cam = np.uint8(result)
#
#                 heatmap = cv2.applyColorMap(np.uint8(255 * refine_cam[0,0].cpu().numpy()), cv2.COLORMAP_JET)
#                 heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#                 result = heatmap * 0.7 + np_image * (1 - 0.7)
#                 result = np.uint8(result)
#                 result = np.concatenate([result_cam,result],axis=1)
#                 plt.imshow(result)
#                 plt.axis('off')
#                 plt.savefig(os.path.join(self.dir_path, image_name.replace(".jpg", cls_name_map[c] + "_refine_cam_{}".format(iik) + ".jpg")),bbox_inches='tight',pad_inches=0,dpi=200)
#
#
#         heatmap = cv2.applyColorMap(np.uint8(255 * score_map[0].cpu().numpy()), cv2.COLORMAP_JET)
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#         result = heatmap * 0.7 + np_image * (1 - 0.7)
#         result = np.uint8(result)
#         plt.imshow(result)
#         plt.axis('off')
#         plt.savefig(os.path.join(self.dir_path, image_name.replace(".jpg", "BG" + ".jpg")))
#
#         heatmap = cv2.applyColorMap(np.uint8(255 * refine_score_map[0].cpu().numpy()), cv2.COLORMAP_JET)
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#         result = heatmap * 0.7 + np_image * (1 - 0.7)
#         result = np.uint8(result)
#         plt.imshow(result)
#         plt.axis('off')
#         plt.savefig(os.path.join(self.dir_path, image_name.replace(".jpg", "BG_refine" + ".jpg")))
#
#         heatmap = cv2.applyColorMap(np.uint8(255 * pseudo_gt[0].cpu().numpy()), cv2.COLORMAP_JET)
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#         result = heatmap * 0.7 + np_image * (1 - 0.7)
#         result = np.uint8(result)
#         plt.imshow(result)
#         plt.axis('off')
#         plt.savefig(os.path.join(self.dir_path, image_name.replace(".jpg", "BG_pseudo_gt" + ".jpg")))
#
#         cams = []
#         for ii,c in enumerate(klasses):
#             cls_prob_tmp = predict_cls[:, c]
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             # heatmap = cv2.applyColorMap(np.uint8(255 * cam.cpu().numpy()), cv2.COLORMAP_JET)
#             # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#             # result = heatmap * 0.7 + image * (1 - 0.7)
#             # result = np.uint8(result)
#             # plt.imshow(result)
#             # plt.axis('off')
#             # plt.savefig(os.path.join(self.dir_path, image_name.replace(".jpg",cls_name_map[c]+".jpg")))
#
#             heatmap = cv2.applyColorMap(np.uint8(255 * score_map[ii+1].cpu().numpy()), cv2.COLORMAP_JET)
#             heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#             result = heatmap * 0.7 + np_image * (1 - 0.7)
#             result = np.uint8(result)
#             plt.imshow(result)
#             plt.axis('off')
#             plt.savefig(os.path.join(self.dir_path, image_name.replace(".jpg", cls_name_map[c] + ".jpg")))
#
#             heatmap = cv2.applyColorMap(np.uint8(255 * refine_score_map[ii+1].cpu().numpy()), cv2.COLORMAP_JET)
#             heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#             result = heatmap * 0.7 + np_image * (1 - 0.7)
#             result = np.uint8(result)
#             plt.imshow(result)
#             plt.axis('off')
#             plt.savefig(os.path.join(self.dir_path, image_name.replace(".jpg", cls_name_map[c] + "_refine" + ".jpg")))
#
#             heatmap = cv2.applyColorMap(np.uint8(255 * pseudo_gt[ii+1].cpu().numpy()), cv2.COLORMAP_JET)
#             heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#             result = heatmap * 0.7 + np_image * (1 - 0.7)
#             result = np.uint8(result)
#             plt.imshow(result)
#             plt.axis('off')
#             plt.savefig(os.path.join(self.dir_path, image_name.replace(".jpg", cls_name_map[c] + "_pseudo_gt.jpg")))
#
#         # start = time.time()
#         # down_scale = 12
#         #
#         # # image = cv2.resize(image,(down_w,down_h),interpolation=cv2.INTER_NEAREST)
#         # image = zoom(image, (1 / down_scale, 1 / down_scale, 1.0))
#         # down_h = image.shape[0]
#         # down_w = image.shape[1]
#         # cams = torch.concat(cams,dim=0)
#         # # cams = cams / (1e-5 + torch.max(cams, dim=0)[0])
#         # cams = torch.nn.functional.interpolate(cams[None,...],[down_h,down_w])[0]
#         # cams = cams.cpu().numpy()
#         # # cams = torch.concat(cams,dim=0).cpu().numpy()
#         #
#         # keys = np.pad(klasses.cpu().numpy() + 1, (1, 0), mode='constant')
#         #
#         # # 1. find confident fg & bg
#         # fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.3)
#         # label_num = fg_conf_cam.shape[0]
#         # fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
#         # pred = crf_inference_label(image, fg_conf_cam, n_labels=label_num)
#         # fg_conf = keys[pred] # mapping 到 klasses 的范围
#         #
#         # bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.05)
#         # bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
#         # pred = crf_inference_label(image, bg_conf_cam, n_labels=label_num)
#         # bg_conf = keys[pred]
#         #
#         # # 2. combine confident fg & bg
#         # # conf 为 最后的 crf 结果 --> label 0: bg, 1: airplane
#         # conf = fg_conf.copy()
#         # # conf[fg_conf == 0] = 255 # ignore
#         # conf[fg_conf == 0] = 21 # ignore
#         # conf[bg_conf + fg_conf == 0] = 0 # bg
#         #
#         # end = time.time()
#         # print(end - start, 's')
#         #
#         # VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
#         #                       (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
#         #                       (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
#         #                       (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255),(200, 200, 200)], np.float32)
#         #
#         # conf_image = VOC_color[conf]
#         # result = 0.7 * conf_image + 0.3 * image
#         # result = np.uint8(result)
#         # plt.imshow(result)
#         # plt.axis('off')
#         # plt.savefig(os.path.join(self.dir_path, image_name))
#
#     # == 0, ignore
#     # != 0 and < 0.25, bg
#     # >= 0.25, fg
#     # < 0.5, complete
#     # >= 0.5, not complete
#     @torch.no_grad()
#     def forward(self, predict_cls, predict_det, rois, labels, iou_map, asy_iou_map,bu_cues, image_name, diffuse=False):
#         if rois.ndim == 3:
#             rois = rois.squeeze(0)
#         rois = rois[:, 1:]  # remove batch_id
#
#         if os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/JPEGImages",image_name)):
#             cls_name_map,cls_id_map = id_2_clsname("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/annotations/voc_2012_val.json")
#
#             image = Image.open(os.path.join("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/JPEGImages",image_name)).convert('RGB')
#             # img
#             s = image_name.replace(".jpg","")
#
#             # proposal
#             COB_proposals = loadmat(os.path.join("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/COB_SBD_trainaug", s + '.mat'))['maskmat'][:, 0]
#
#             mask_proposals = [np.array(p) for p in COB_proposals]
#             masks = np.array(mask_proposals)
#
#         # COCO
#         elif os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/train2017",image_name)):
#             cls_name_map,cls_id_map = id_2_clsname("/home/lzc/WSIS-Benchmark/dataset/coco2017/annotations/instances_val2017.json")
#
#             image = Image.open(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/train2017",image_name)).convert('RGB')
#             # proposal
#             s = image_name.replace(".jpg","")
#
#             file_name = 'COCO_train2014_' + s + '.mat'
#             if not os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/COB-COCO", file_name)):
#                 file_name = 'COCO_val2014_' + s + '.mat'
#             if not os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/COB-COCO", file_name)):
#                 file_name = s + '.mat'
#             COB_proposals = loadmat(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/COB-COCO", file_name),
#                                              verify_compressed_data_integrity=False)['maskmat']
#
#             mask_proposals = [np.array(p) for p in COB_proposals]
#             masks = np.array(mask_proposals)
#         else:
#             raise AssertionError
#         if diffuse:
#             self.mist_label_diffuse(predict_cls, predict_det,
#                                      rois, labels, iou_map,
#                                      asy_iou_map,bu_cues,masks,image,cls_name_map,image_name)
#         else:
#             raise AssertionError

# 将 proposal 叠起来，可视化 heatmap
# grad_CAM
class mist_layer_visual_heatmap(nn.Module):
    def __init__(self, portion=0.1, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85, dir_path=None,PAMR_model=None,cam_extracter=None):
        super(mist_layer_visual_heatmap, self).__init__()
        self.portion = portion
        self.full_thr = full_thr
        self.iou_th = iou_thr
        self.asy_iou_th = asy_iou_th
        self.dir_path = dir_path

        print("mist_layer_visual_heatmap--> full_thr: {}, iou_thr: {}".format(full_thr, iou_thr))
        print("dir path: {}".format(dir_path))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.PAMR_model = PAMR_model
        self.cam_extracter = cam_extracter

    def pseudo_gtmask(self, mask, cutoff_top=0.6, cutoff_low=0.2, eps=1e-8):
        """Convert continuous mask into binary mask"""
        bs,c,h,w = mask.size()
        mask = mask.view(bs,c,-1)

        # for each class extract the max confidence
        mask_max, _ = mask.max(-1, keepdim=True)
        mask_max[:, :1] *= 0.7
        mask_max[:, 1:] *= cutoff_top
        #mask_max *= cutoff_top

        # if the top score is too low, ignore it
        # lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
        # mask_max = mask_max.max(lowest)

        pseudo_gt = (mask > mask_max).type_as(mask)

        # remove ambiguous pixels
        ambiguous = (pseudo_gt.sum(1, keepdim=True) > 1).type_as(mask)
        pseudo_gt = (1 - ambiguous) * pseudo_gt

        return pseudo_gt.view(bs,c,h,w)

    # instance_list -> [{},{}...]
    # {} -> {score: float, mask_id: int}
    def instance_nms(self, instance_list, iou_map):
        instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)

        selected_instances_id = []
        group_instances_id = []
        while len(instance_list) > 0:
            src_instance = instance_list.pop(0)
            selected_instances_id.append(src_instance["mask_id"])

            src_mask_id = src_instance["mask_id"]

            keep = []
            drop = []
            drop.append(src_mask_id)

            for dst_instance in instance_list:
                dst_mask_id = dst_instance["mask_id"]

                iou = iou_map[src_mask_id][dst_mask_id]
                if iou < self.iou_th:
                    keep.append(dst_instance)

                else:
                    drop.append(dst_mask_id)

            instance_list = keep
            group_instances_id.append(drop)

        return selected_instances_id,group_instances_id

    @torch.no_grad()
    def mist_label(self, preds, rois, label, iou_map=None, asy_iou_map=None):
        if label.dim() != 1:
            label = label.squeeze()

        assert label.dim() == 1
        assert label.shape[-1] == 20 or label.shape[-1] == 80

        # bg remove
        preds = (preds if preds.shape[-1] == label.shape[-1] else preds[:, 1:]).clone()  # remove background class if present
        keep_count = int(np.ceil(self.portion * preds.shape[0]))
        klasses = label.nonzero(as_tuple=True)[0]
        # one hot label
        gt_labels = torch.zeros((preds.shape[0], label.shape[-1] + 1), dtype=preds.dtype, device=preds.device)
        gt_weights = -torch.ones((preds.shape[0],), dtype=preds.dtype, device=preds.device)

        for c in klasses:
            cls_prob_tmp = preds[:, c]

            keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals

            keep_rois = rois[keep_sort_idx]
            keep_cls_prob = cls_prob_tmp[keep_sort_idx]

            # iou nms
            if iou_map != None:
                temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]

                instance_list = []
                for i, prob in enumerate(keep_cls_prob):
                    instance = dict()

                    instance["score"] = prob
                    instance["mask_id"] = i
                    instance_list.append(instance)

                keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
                keep_nms_idx = torch.tensor(keep_nms_idx,device=preds.device)

            # box nms
            else:
                print("iou_map == None")
                keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)

            keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index

            is_higher_scoring_class = cls_prob_tmp[keep_nms_idx] > gt_weights[keep_nms_idx]
            keep_idxs = keep_nms_idx[is_higher_scoring_class]
            gt_labels[keep_idxs, :] = 0
            gt_labels[keep_idxs, c + 1] = 1
            gt_weights[keep_idxs] = cls_prob_tmp[keep_idxs]

        gt_idxs = torch.sum(gt_labels, dim=-1) > 0

        gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]

        return gt_boxes, gt_labels, gt_weights, gt_idxs

    def mist_label_diffuse(self, predict_cls, predict_det, rois, label, iou_map, asy_iou_map, bu_cues,masks,image,cls_name_map,image_name):
        if label.dim() != 1:
            label = label.squeeze()
        self.down_scale = 1

        assert label.dim() == 1
        assert label.shape[-1] == 20 or label.shape[-1] == 80
        image = torch.tensor(np.array(image),dtype=torch.float32,device=predict_cls.device)

        image = torch.nn.functional.interpolate(image.permute(2, 0, 1)[None, ...],
                                                scale_factor=(self.down_scale, self.down_scale),
                                                mode="bilinear", align_corners=True, recompute_scale_factor=True)


        masks = torch.tensor(masks, dtype=torch.float32, device=predict_cls.device)
        masks = torch.nn.functional.interpolate(masks[:, None], scale_factor=(self.down_scale, self.down_scale),
                                                    mode="bilinear", align_corners=True, recompute_scale_factor=True)

        # bg remove
        predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
        predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present

        preds = predict_cls * predict_det

        # keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))

        klasses = label.nonzero(as_tuple=True)[0]
        # one hot label
        gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
        gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)

        # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
        asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]

        for c in klasses:
            cls_prob_tmp = predict_cls[:, c]
            if predict_det.shape[-1] == label.shape[-1]:
                det_prob_tmp = predict_det[:, c]
            elif predict_det.shape[-1] == 1:
                det_prob_tmp = predict_det[:, 0]
            else:
                raise AssertionError

            preds_tmp = preds[:, c]

            keep_count = min(int(torch.sum(cls_prob_tmp >= (torch.max(cls_prob_tmp) * 0.3))),
                             int(np.ceil(self.portion * predict_cls.shape[0])))
            keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals

            keep_rois = rois[keep_sort_idx]
            keep_cls_prob = cls_prob_tmp[keep_sort_idx]

            # cal iou nms
            if iou_map != None:
                temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]

                instance_list = []
                for i, prob in enumerate(keep_cls_prob):
                    instance = dict()

                    instance["score"] = prob
                    instance["mask_id"] = i
                    instance_list.append(instance)

                keep_nms_idx,group_instances_ids = self.instance_nms(instance_list, temp_iou_map)
                keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)

            # box nms
            else:
                raise AssertionError

            keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
            cams = self.cam_extracter([c for _ in range(predict_cls.shape[0])], predict_cls, normalized=False, retain_graph=True)[0]
            print(cams.shape)
            for iik, group_instances_id in enumerate(group_instances_ids):
                print(group_instances_id)
                real_group_id = keep_sort_idx[torch.tensor(group_instances_id, device=predict_cls.device)]
                temp_mask = torch.sum(masks[real_group_id],dim=(0,1))
                temp_mask /= torch.max(temp_mask)
                # scores = predict_cls[real_group_id]
                # scores = torch.mean(scores,dim=0,keepdim=True)

                cam = cams[real_group_id]
                cam = cam.sum(0)

                cam = cam - torch.min(cam)
                cam /= (1e-5 + torch.max(cam))
                cam = torch.nn.functional.interpolate(cam[None,None,...], size=image.shape[-2:],
                                                       mode="bilinear", align_corners=True)

                refine_cam = self.PAMR_model(image, cam)
                refine_cam /= (1e-5 + torch.max(refine_cam))

                heatmap = cv2.applyColorMap(np.uint8(255 * temp_mask.cpu().numpy()), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                result = heatmap * 0.7 + image[0].permute(1, 2, 0).cpu().numpy() * (1 - 0.7)
                result_mask = np.uint8(result)

                heatmap = cv2.applyColorMap(np.uint8(255 * cam[0,0].cpu().numpy()), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                result = heatmap * 0.7 + image[0].permute(1, 2, 0).cpu().numpy() * (1 - 0.7)
                result_cam = np.uint8(result)

                heatmap = cv2.applyColorMap(np.uint8(255 * refine_cam[0,0].cpu().numpy()), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                result = heatmap * 0.7 + image[0].permute(1, 2, 0).cpu().numpy() * (1 - 0.7)
                result = np.uint8(result)
                result = np.concatenate([result_mask,result_cam,result],axis=1)
                plt.imshow(result)
                plt.axis('off')
                plt.savefig(os.path.join(self.dir_path, image_name.replace(".jpg", cls_name_map[c] + "_refine_cam_{}".format(iik) + ".jpg")),bbox_inches='tight',pad_inches=0,dpi=200)

    # == 0, ignore
    # != 0 and < 0.25, bg
    # >= 0.25, fg
    # < 0.5, complete
    # >= 0.5, not complete
    def forward(self, predict_cls, predict_det, rois, labels, iou_map, asy_iou_map,bu_cues, image_name, diffuse=False):
        if rois.ndim == 3:
            rois = rois.squeeze(0)
        rois = rois[:, 1:]  # remove batch_id

        if os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/JPEGImages",image_name)):
            cls_name_map,cls_id_map = id_2_clsname("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/annotations/voc_2012_val.json")

            image = Image.open(os.path.join("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/JPEGImages",image_name)).convert('RGB')
            # img
            s = image_name.replace(".jpg","")

            # proposal
            COB_proposals = loadmat(os.path.join("/home/lzc/WSIS-Benchmark/dataset/VOCdevkit/VOC2012/COB_SBD_trainaug", s + '.mat'))['maskmat'][:, 0]

            mask_proposals = [np.array(p) for p in COB_proposals]
            masks = np.array(mask_proposals)

        # COCO
        elif os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/train2017",image_name)):
            cls_name_map,cls_id_map = id_2_clsname("/home/lzc/WSIS-Benchmark/dataset/coco2017/annotations/instances_val2017.json")

            image = Image.open(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/train2017",image_name)).convert('RGB')
            # proposal
            s = image_name.replace(".jpg","")

            file_name = 'COCO_train2014_' + s + '.mat'
            if not os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/COB-COCO", file_name)):
                file_name = 'COCO_val2014_' + s + '.mat'
            if not os.path.exists(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/COB-COCO", file_name)):
                file_name = s + '.mat'
            COB_proposals = loadmat(os.path.join("/home/lzc/WSIS-Benchmark/dataset/coco2017/COB-COCO", file_name),
                                             verify_compressed_data_integrity=False)['maskmat']

            mask_proposals = [np.array(p) for p in COB_proposals]
            masks = np.array(mask_proposals)
        else:
            raise AssertionError
        if diffuse:
            self.mist_label_diffuse(predict_cls, predict_det,
                                     rois, labels, iou_map,
                                     asy_iou_map,bu_cues,masks,image,cls_name_map,image_name,rois)
        else:
            raise AssertionError

# # 对 使用 pamr 模块对网络进行 guidance
# class mist_layer_with_pamr(nn.Module):
#     def __init__(self, portion=0.1, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85,PAMR_model=None):
#         super(mist_layer_with_pamr, self).__init__()
#         self.portion = portion
#         self.full_thr = full_thr
#         self.iou_th = iou_thr
#         self.asy_iou_th = asy_iou_th
#         self.PAMR_model = PAMR_model
#
#         print("mist_layer_with_pamr--> portion:{}, full_thr: {}, iou_thr: {}".format(portion,full_thr,iou_thr))
#
#     def pseudo_gtmask(self, mask, cutoff_top=0.6, cutoff_low=0.2, eps=1e-8):
#         """Convert continuous mask into binary mask"""
#         bs,c,h,w = mask.size()
#         mask = mask.view(bs,c,-1)
#
#         # for each class extract the max confidence
#         mask_max, _ = mask.max(-1, keepdim=True)
#         mask_max[:, :1] *= 0.7
#         mask_max[:, 1:] *= cutoff_top
#         #mask_max *= cutoff_top
#
#         # if the top score is too low, ignore it
#         # lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
#         # mask_max = mask_max.max(lowest)
#
#         pseudo_gt = (mask > mask_max).type_as(mask)
#
#         # remove ambiguous pixels
#         ambiguous = (pseudo_gt.sum(1, keepdim=True) > 1).type_as(mask)
#         pseudo_gt = (1 - ambiguous) * pseudo_gt
#
#         return pseudo_gt.view(bs,c,h,w)
#
#     # # 对 image 进行下采样以后 使用 pamr 对 mask 进行 refine
#     # def run_pamr(self, im, mask):
#     #     im = F.interpolate(im, mask.size()[-2:], mode="bilinear", align_corners=True)
#     #     masks_dec = self.PAMR_model(im, mask)
#     #     return masks_dec
#
#     # instance_list -> [{},{}...]
#     # {} -> {score: float, mask_id: int}
#     def instance_nms(self, instance_list, iou_map):
#         instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)
#
#         selected_instances_id = []
#         while len(instance_list) > 0:
#             src_instance = instance_list.pop(0)
#             selected_instances_id.append(src_instance["mask_id"])
#
#             src_mask_id = src_instance["mask_id"]
#
#             def iou_filter(dst_instance):
#                 dst_mask_id = dst_instance["mask_id"]
#
#                 iou = iou_map[src_mask_id][dst_mask_id]
#                 if iou < self.iou_th:
#                     return True
#                 else:
#                     return False
#
#             instance_list = list(filter(iou_filter, instance_list))
#
#         return selected_instances_id
#
#     # instance_list -> [{},{}...]
#     # {} -> {score: float, mask_id: int}
#     def instance_asy_nms(self, instance_list, asy_iou_map):
#         instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)
#
#         selected_instances_id = []
#         while len(instance_list) > 0:
#             src_instance = instance_list.pop(0)
#             selected_instances_id.append(src_instance["mask_id"])
#
#             src_mask_id = src_instance["mask_id"]
#
#             def iou_filter(dst_instance):
#                 dst_mask_id = dst_instance["mask_id"]
#
#                 iou = asy_iou_map[src_mask_id][dst_mask_id]
#                 if iou < self.asy_iou_th:
#                     return True
#                 else:
#                     return False
#
#             instance_list = list(filter(iou_filter, instance_list))
#
#         return selected_instances_id
#
#     @torch.no_grad()
#     def mist_label(self, preds, rois, label, iou_map=None, asy_iou_map=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         preds = (preds if preds.shape[-1] == label.shape[-1] else preds[:, 1:]).clone()  # remove background class if present
#         keep_count = int(np.ceil(self.portion * preds.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((preds.shape[0], label.shape[-1] + 1), dtype=preds.dtype, device=preds.device)
#         gt_weights = -torch.ones((preds.shape[0],), dtype=preds.dtype, device=preds.device)
#
#         for c in klasses:
#             cls_prob_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=preds.device)
#
#             # box nms
#             else:
#                 print("iou_map == None")
#                 keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             is_higher_scoring_class = cls_prob_tmp[keep_nms_idx] > gt_weights[keep_nms_idx]
#             keep_idxs = keep_nms_idx[is_higher_scoring_class]
#             gt_labels[keep_idxs, :] = 0
#             gt_labels[keep_idxs, c + 1] = 1
#             gt_weights[keep_idxs] = cls_prob_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs
#
#     # proposal 在每张图片处都只 rescale 一次
#     @torch.no_grad()
#     def mist_label_diffuse(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None, image=None,proposals=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         bg_prob = predict_cls[:, 0][:,None]
#
#         # bg remove
#         predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
#         predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present
#
#         preds = predict_cls * predict_det
#
#         keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
#         gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)
#
#         # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
#         asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]
#
#         fg_prob = predict_cls[:,klasses]
#         cam = torch.concat((bg_prob,fg_prob),dim=1) # 200,sum(labels)+1
#
#         # score_map = (proposals * cam[...,None,None]).sum(0) # klasses.size() + 1
#         score_map = (proposals * cam[..., None, None]).sum(0) / (1e-5 + torch.tensor(proposals, device=cam.device).sum(0))  # klasses.size() + 1
#         b,h,w = score_map.shape
#
#         score_map = score_map.view(b,-1)
#         min_score = torch.min(score_map,dim=-1,keepdim=True)[0]
#         max_score = torch.max(score_map,dim=-1,keepdim=True)[0]
#
#         score_map = (score_map - min_score) / (max_score - min_score + 1e-5)
#         refine_score_map = self.PAMR_model(image, score_map.view(1,b,h,w))
#         pseudo_gt = self.pseudo_gtmask(refine_score_map)
#
#         for class_idx, c in enumerate(klasses):
#             cost = -1 * torch.ones(len(klasses)+1,device=pseudo_gt.device)
#             cost[class_idx + 1] = 1
#
#             cls_prob_tmp = predict_cls[:, c]
#             if predict_det.shape[-1] == label.shape[-1]:
#                 det_prob_tmp = predict_det[:, c]
#             elif predict_det.shape[-1] == 1:
#                 det_prob_tmp = predict_det[:, 0]
#             else:
#                 raise AssertionError
#
#             preds_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # cal iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)
#
#             # box nms
#             else:
#                 print("iou_map == None")
#                 keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             assert asy_iou_map != None
#             temp_asy_iou_map = asy_iou_map[:, keep_nms_idx]
#             temp_asy_iou_map = temp_asy_iou_map > self.asy_iou_th
#
#             flag = temp_asy_iou_map * asy_iou_flag
#             org_det = det_prob_tmp[keep_nms_idx]
#
#             if flag.sum() != 0:
#                 col_idxs = torch.sum(flag,dim=0) > 0
#                 flag = flag[:, col_idxs]
#                 org_det = org_det[col_idxs][None,:] # 1, N
#                 keep_nms_idx = keep_nms_idx[col_idxs]
#
#                 res_det = flag * det_prob_tmp[:,None]
#
#                 diffuse_idxs = res_det > org_det
#                 diffuse_flags = torch.sum(diffuse_idxs,dim=0) > 0
#
#                 res_idx = []
#                 for idx, diffuse_flag in enumerate(diffuse_flags):
#                     if diffuse_flag:
#                         proposal_scores = torch.sum(proposals[diffuse_idxs[:, idx]] * pseudo_gt * cost[None,:,None,None],dim=(-1,-2,-3))
#                         max_idx = torch.argmax(proposal_scores) # 0-sum(diffuse_idxs[:, idx])
#                         max_idx = torch.nonzero(diffuse_idxs[:, idx])[max_idx] # 0-200 # mapping to real index
#                         res_idx.append(max_idx)
#
#                     else:
#                         res_idx.append(keep_nms_idx[idx])
#
#                 res_idx = torch.tensor(res_idx) # 真正的 gt
#                 res_idx = torch.unique(res_idx)
#
#                 is_higher_scoring_class = preds_tmp[res_idx] > gt_weights[res_idx]
#                 if is_higher_scoring_class.sum() > 0:
#                     keep_idxs = res_idx[is_higher_scoring_class]
#                     gt_labels[keep_idxs, :] = 0
#                     gt_labels[keep_idxs, c + 1] = 1
#                     gt_weights[keep_idxs] = preds_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#         # assert gt_idxs.sum() > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag
#
#     # == 0, ignore
#     # != 0 and < 0.25, bg
#     # >= 0.25, fg
#     # < 0.5, complete
#     # >= 0.5, not complete
#     @torch.no_grad()
#     def forward(self, predict_cls, predict_det, rois, labels, iou_map=None, asy_iou_map=None, diffuse=False,image=None,proposals=None):
#         if rois.ndim == 3:
#             rois = rois.squeeze(0)
#         rois = rois[:, 1:]  # remove batch_id
#
#         if diffuse:
#             gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag = self.mist_label_diffuse(predict_cls, predict_det,
#                                                                                              rois, labels, iou_map,
#                                                                                              asy_iou_map,image,proposals)
#         else:
#             if predict_det != None:
#                 preds = predict_cls * predict_det
#             else:
#                 preds = predict_cls
#
#             gt_boxes, gt_labels, gt_weights, gt_idxs = self.mist_label(preds, rois, labels, iou_map, asy_iou_map)
#
#         if gt_idxs.sum() == 0:
#             return None, None, None, None
#
#         if iou_map == None:
#             overlaps = box_iou(rois, gt_boxes)  # (proposal_num,gt_num)
#         else:
#             overlaps = iou_map[:, gt_idxs]
#
#         max_overlap_v, max_overlap_idx = torch.max(overlaps, dim=-1)
#
#         pseudo_labels = gt_labels[max_overlap_idx]
#         loss_weights = gt_weights[max_overlap_idx]
#         pseudo_iou_label = max_overlap_v
#
#         ignore_inds = max_overlap_v == 0
#         pseudo_labels[ignore_inds, :] = 0
#         loss_weights[ignore_inds] = 0
#
#         bg_inds = (max_overlap_v < self.iou_th) * ~ignore_inds
#         pseudo_labels[bg_inds, :] = 0
#         pseudo_labels[bg_inds, 0] = 1
#
#         try:
#             # 将太大的proposal标记为bg
#             big_proposal = ~asy_iou_flag
#             pseudo_labels[big_proposal, :] = 0
#             pseudo_labels[big_proposal, 0] = 1
#         except:
#             pass
#
#         pseudo_iou_label[pseudo_iou_label > self.full_thr] = 1
#         pseudo_iou_label[pseudo_iou_label <= self.full_thr] = 0
#
#         # pseudo_iou_label = (pseudo_iou_label - self.iou_th) / (1 - self.iou_th)
#         # pseudo_iou_label = (pseudo_iou_label - 0.5) / (1 - 0.5)
#
#         group_assign = max_overlap_idx + 1  # 将 index 从 1 开始
#         group_assign[bg_inds] = -1
#         group_assign[ignore_inds] = -2
#         # 所以最后的范围是 [-2, -1, , 1, 2, 3...]
#
#         # transform pseudo_labels --> N * 21
#         # 将 group_assign 变为 和 初始伪标签 mat 相同的形态
#         group_assign = group_assign[:, None] * pseudo_labels
#
#         return pseudo_labels, pseudo_iou_label, loss_weights, group_assign

# # 对 使用 pamr 模块对网络进行 guidance
# class mist_layer_with_pamr(mist_layer):
#     def __init__(self, portion=0.1, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85,PAMR_model=None,sample=False,test_mode=False,get_diffuse_gt=False):
#         super().__init__(portion, full_thr, iou_thr, asy_iou_th,sample,test_mode,get_diffuse_gt)
#         self.PAMR_model = PAMR_model
#
#         print("mist_layer_with_pamr--> portion:{}, full_thr: {}, iou_thr: {}".format(portion,full_thr,iou_thr))
#
#     def pseudo_gtmask(self, mask, cutoff_top=0.6, cutoff_low=0.2, eps=1e-8):
#         """Convert continuous mask into binary mask"""
#         bs,c,h,w = mask.size()
#         mask = mask.view(bs,c,-1)
#
#         # for each class extract the max confidence
#         mask_max, _ = mask.max(-1, keepdim=True)
#         mask_max[:, :1] *= 0.7
#         mask_max[:, 1:] *= cutoff_top
#         #mask_max *= cutoff_top
#
#         # if the top score is too low, ignore it
#         # lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
#         # mask_max = mask_max.max(lowest)
#
#         pseudo_gt = (mask > mask_max).type_as(mask)
#
#         # remove ambiguous pixels
#         ambiguous = (pseudo_gt.sum(1, keepdim=True) > 1).type_as(mask)
#         pseudo_gt = (1 - ambiguous) * pseudo_gt
#
#         return pseudo_gt.view(bs,c,h,w)
#
#     def pseudo_gtmask_full(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None, image=None,proposals=None):
#         assert image != None
#         assert proposals != None
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         bg_prob = predict_cls[:, 0][:, None]
#
#         # bg remove
#         predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1] - 1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
#
#         klasses = label.nonzero(as_tuple=True)[0]
#
#         fg_prob = predict_cls[:, klasses]
#         cam = torch.concat((bg_prob, fg_prob), dim=1)  # 200,sum(labels)+1
#
#         score_map = (proposals * cam[..., None, None]).sum(0) \
#                     / (1e-5 + proposals.sum(0))  # klasses.size() + 1
#         b, h, w = score_map.shape
#
#         refine_score_map = self.PAMR_model(image, score_map.view(1, b, h, w))
#
#         pseudo_gt = self.pseudo_gtmask(refine_score_map)
#
#         return pseudo_gt
#
#     # # 返回 score
#     # def pseudo_gtmask_score(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None, image=None,proposals=None):
#     #     if label.dim() != 1:
#     #         label = label.squeeze()
#     #
#     #     assert label.dim() == 1
#     #     assert label.shape[-1] == 20 or label.shape[-1] == 80
#     #
#     #     bg_prob = predict_cls[:, 0][:, None]
#     #
#     #     # bg remove
#     #     predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1] - 1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
#     #
#     #     klasses = label.nonzero(as_tuple=True)[0]
#     #
#     #     fg_prob = predict_cls[:, klasses]
#     #     cam = torch.concat((bg_prob, fg_prob), dim=1)  # 200,sum(labels)+1
#     #
#     #     score_map = (proposals * cam[..., None, None]).sum(0) \
#     #                 / (1e-5 + proposals.sum(0))  # klasses.size() + 1
#     #     b, h, w = score_map.shape
#     #
#     #     refine_score_map = self.PAMR_model(image, score_map.view(1, b, h, w))
#     #
#     #     pseudo_gt = self.pseudo_gtmask(refine_score_map) # 1,klasses.size() + 1,H,w
#     #
#     #     pseudo_gt_score = (proposals * pseudo_gt).sum(dim=(-1,-2)) # 200,klasses.size() + 1
#     #
#     #     return pseudo_gt_score
#
#
#     # proposal 在每张图片处都只 rescale 一次
#     # SS score
#
#     # 返回 group 信息
#     def instance_nms_with_group(self, instance_list, iou_map):
#         instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)
#
#         selected_instances_id = []
#         group_instances_id = []
#         while len(instance_list) > 0:
#             src_instance = instance_list.pop(0)
#             selected_instances_id.append(src_instance["mask_id"])
#
#             src_mask_id = src_instance["mask_id"]
#
#             keep = []
#             drop = []
#             drop.append(src_mask_id)
#
#             for dst_instance in instance_list:
#                 dst_mask_id = dst_instance["mask_id"]
#
#                 iou = iou_map[src_mask_id][dst_mask_id]
#                 if iou < self.iou_th:
#                     keep.append(dst_instance)
#
#                 else:
#                     drop.append(dst_mask_id)
#
#             instance_list = keep
#             group_instances_id.append(drop)
#
#         return selected_instances_id, group_instances_id
#
#     @torch.no_grad()
#     def mist_label_diffuse_SS(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None, image=None,proposals=None, pseudo_gt=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         bg_prob = predict_cls[:, 0][:,None]
#
#         # bg remove
#         predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
#         predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present
#
#         preds = predict_cls * predict_det
#
#         keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
#         gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)
#
#         # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
#         asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]
#
#         if pseudo_gt == None:
#             fg_prob = predict_cls[:,klasses]
#             cam = torch.concat((bg_prob,fg_prob),dim=1) # 200,sum(labels)+1
#
#             # score_map = (proposals * cam[...,None,None]).sum(0) # klasses.size() + 1
#             score_map = (proposals * cam[..., None, None]).sum(0) \
#                         / (1e-5 + proposals.sum(0))  # klasses.size() + 1
#             b,h,w = score_map.shape
#
#             # score_map = score_map.view(b,-1)
#             # min_score = torch.min(score_map,dim=-1,keepdim=True)[0]
#             # max_score = torch.max(score_map,dim=-1,keepdim=True)[0]
#
#             # score_map = (score_map - min_score) / (max_score - min_score + 1e-5)
#
#             refine_score_map = self.PAMR_model(image, score_map.view(1,b,h,w))
#
#             pseudo_gt = self.pseudo_gtmask(refine_score_map)
#
#         for class_idx, c in enumerate(klasses):
#             cost = -1 * torch.ones(len(klasses)+1,device=pseudo_gt.device)
#             cost[class_idx + 1] = 1
#
#             cls_prob_tmp = predict_cls[:, c]
#             if predict_det.shape[-1] == label.shape[-1]:
#                 det_prob_tmp = predict_det[:, c]
#             elif predict_det.shape[-1] == 1:
#                 det_prob_tmp = predict_det[:, 0]
#             else:
#                 raise AssertionError
#
#             preds_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # cal iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)
#
#             # box nms
#             else:
#                 print("iou_map == None")
#                 keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             assert asy_iou_map != None
#             temp_asy_iou_map = asy_iou_map[:, keep_nms_idx]
#             temp_asy_iou_map = temp_asy_iou_map > self.asy_iou_th
#
#             flag = temp_asy_iou_map * asy_iou_flag
#             org_det = det_prob_tmp[keep_nms_idx]
#
#             if flag.sum() != 0:
#                 if self.test_mode and not self.get_diffuse_gt:
#                     res_idx = keep_nms_idx[torch.sum(flag, dim=0) > 0]  # 原先的 res_idx --> 未diffuse
#                 else:
#                     col_idxs = torch.sum(flag,dim=0) > 0
#                     flag = flag[:, col_idxs]
#                     org_det = org_det[col_idxs][None,:] # 1, N
#                     keep_nms_idx = keep_nms_idx[col_idxs]
#
#                     res_det = flag * det_prob_tmp[:,None]
#
#                     diffuse_idxs = res_det > org_det
#                     diffuse_flags = torch.sum(diffuse_idxs,dim=0) > 0
#
#                     res_idx = []
#                     for idx, diffuse_flag in enumerate(diffuse_flags):
#                         if diffuse_flag:
#                             proposal_scores = torch.sum(proposals[diffuse_idxs[:, idx]] * pseudo_gt * cost[None,:,None,None],dim=(-1,-2,-3)) / torch.sum(proposals[diffuse_idxs[:, idx]],dim=(-1,-2,-3))
#                             max_idx = torch.argmax(proposal_scores) # 0-sum(diffuse_idxs[:, idx])
#                             max_idx = torch.nonzero(diffuse_idxs[:, idx])[max_idx] # 0-200 # mapping to real index
#                             res_idx.append(max_idx)
#
#                         else:
#                             res_idx.append(keep_nms_idx[idx])
#
#                     res_idx = torch.tensor(res_idx) # 真正的 gt
#                     res_idx = torch.unique(res_idx)
#
#                 is_higher_scoring_class = preds_tmp[res_idx] > gt_weights[res_idx]
#                 if is_higher_scoring_class.sum() > 0:
#                     keep_idxs = res_idx[is_higher_scoring_class]
#                     gt_labels[keep_idxs, :] = 0
#                     gt_labels[keep_idxs, c + 1] = 1
#                     gt_weights[keep_idxs] = preds_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#         # assert gt_idxs.sum() > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag
#
#     # SS score forward
#     @torch.no_grad()
#     def SS_forward(self, predict_cls, predict_det, rois, labels, iou_map=None, asy_iou_map=None, diffuse=False,image=None,proposals=None, pseudo_gt=None):
#         if rois.ndim == 3:
#             rois = rois.squeeze(0)
#         rois = rois[:, 1:]  # remove batch_id
#
#         if diffuse:
#             gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag = self.mist_label_diffuse_SS(predict_cls, predict_det,
#                                                                                              rois, labels, iou_map,
#                                                                                              asy_iou_map,image,proposals,pseudo_gt=pseudo_gt)
#         else:
#             if predict_det != None:
#                 preds = predict_cls * predict_det
#             else:
#                 preds = predict_cls
#
#             gt_boxes, gt_labels, gt_weights, gt_idxs = self.mist_label(preds, rois, labels, iou_map, asy_iou_map)
#
#         if gt_idxs.sum() == 0:
#             return None, None, None, None
#
#         if iou_map == None:
#             overlaps = box_iou(rois, gt_boxes)  # (proposal_num,gt_num)
#         else:
#             overlaps = iou_map[:, gt_idxs]
#
#         # # add sample mode
#         # # 对 gt 进行 sample
#         # # 消除面积对 proposal 的 bias
#         #
#         # # False --> 清除位
#         # # True  --> 保持位
#         # inds = torch.ones_like(gt_labels[:, 0], device=gt_labels.device)
#         # if labels.dim() != 1:
#         #     label = labels.squeeze()
#         # else:
#         #     label = labels
#         #
#         # assert label.dim() == 1
#         # assert label.shape[-1] == 20 or label.shape[-1] == 80
#         #
#         # klasses = label.nonzero(as_tuple=True)[0]
#         #
#         # for c in klasses:
#         #     class_idx = torch.nonzero(gt_labels[:, c + 1] == 1).flatten().cpu().numpy()
#         #     if len(class_idx) == 0:
#         #         continue
#         #
#         #     prob = gt_weights[class_idx].cpu().numpy()
#         #     sampled_class_idx = np.random.choice(class_idx, size=len(class_idx), replace=True,
#         #                                          p=prob / prob.sum())
#         #     sampled_class_idx = np.unique(sampled_class_idx)
#         #
#         #     inds[class_idx] = 0
#         #     inds[sampled_class_idx] = 1
#         #
#         # inds = inds == 1
#         # gt_weights = gt_weights[inds]
#         # gt_labels = gt_labels[inds, :]
#         # gt_boxes = gt_boxes[inds, :]
#         # overlaps = overlaps[:, inds]
#         #
#         # # sample done
#         # ################
#
#         max_overlap_v, max_overlap_idx = torch.max(overlaps, dim=-1)
#
#         pseudo_labels = gt_labels[max_overlap_idx]
#         loss_weights = gt_weights[max_overlap_idx]
#         pseudo_iou_label = max_overlap_v
#
#         if self.test_mode:
#             bg_inds = (max_overlap_v != 1)
#             pseudo_labels[bg_inds,:] = 0
#             pseudo_labels[bg_inds,0] = 1
#             pseudo_labels = pseudo_labels * loss_weights[:,None]
#
#             return pseudo_labels, pseudo_iou_label, loss_weights, None
#
#         else:
#             ignore_inds = max_overlap_v == 0
#             pseudo_labels[ignore_inds, :] = 0
#             loss_weights[ignore_inds] = 0
#
#             bg_inds = (max_overlap_v < self.iou_th) * ~ignore_inds
#             pseudo_labels[bg_inds, :] = 0
#             pseudo_labels[bg_inds, 0] = 1
#
#             try:
#                 # 将太大的proposal标记为bg
#                 big_proposal = ~asy_iou_flag
#                 pseudo_labels[big_proposal, :] = 0
#                 pseudo_labels[big_proposal, 0] = 1
#             except:
#                 pass
#
#             pseudo_iou_label[pseudo_iou_label > self.full_thr] = 1
#             pseudo_iou_label[pseudo_iou_label <= self.full_thr] = 0
#
#             # pseudo_iou_label = (pseudo_iou_label - self.iou_th) / (1 - self.iou_th)
#             # pseudo_iou_label = (pseudo_iou_label - 0.5) / (1 - 0.5)
#
#             group_assign = max_overlap_idx + 1  # 将 index 从 1 开始
#             group_assign[bg_inds] = -1
#             group_assign[ignore_inds] = -2
#             # 所以最后的范围是 [-2, -1, , 1, 2, 3...]
#
#             # transform pseudo_labels --> N * 21
#             # 将 group_assign 变为 和 初始伪标签 mat 相同的形态
#             group_assign = group_assign[:, None] * pseudo_labels
#
#             return pseudo_labels, pseudo_iou_label, loss_weights, group_assign
#
#     # IS score
#     @torch.no_grad()
#     def mist_label_diffuse_IS(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None, image=None,proposals=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
#         predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present
#
#         preds = predict_cls * predict_det
#
#         keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
#         gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)
#
#         # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
#         asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]
#
#         for class_idx, c in enumerate(klasses):
#             cls_prob_tmp = predict_cls[:, c]
#             if predict_det.shape[-1] == label.shape[-1]:
#                 det_prob_tmp = predict_det[:, c]
#             elif predict_det.shape[-1] == 1:
#                 det_prob_tmp = predict_det[:, 0]
#             else:
#                 raise AssertionError
#
#             preds_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # cal iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx, group_instances_idx = self.instance_nms_with_group(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)
#
#             # box nms
#             else:
#                 raise AssertionError
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             # no longer sample diffuse area
#             # sample seed area
#             # --> easy_case_mining
#             if self.sample:
#                 prob = preds_tmp[keep_nms_idx].cpu().numpy()
#                 # keep_nms_idx = keep_nms_idx.cpu().numpy()
#                 keep_index = np.arange(len(keep_nms_idx)) # start from zero
#                 sampled_class_idx = np.random.choice(keep_index, size=len(keep_index), replace=True,
#                                                      p=prob / prob.sum())
#                 keep_index = torch.tensor(np.unique(sampled_class_idx),device=predict_cls.device)
#                 keep_nms_idx = keep_nms_idx[keep_index]
#
#                 store = []
#                 for idx, group_instances_id in enumerate(group_instances_idx):
#                     if idx in keep_index:
#                         store.append(group_instances_id)
#
#                 group_instances_idx = store
#
#             group_instances_idxs = torch.zeros(len(proposals),len(keep_nms_idx),device=label.device) # 200, xxx
#             for idx, group_instances_id in enumerate(group_instances_idx):
#                 group_instances_idxs[keep_sort_idx[group_instances_id],idx] = 1
#
#             group_instances_idxs = group_instances_idxs == 1
#
#             assert asy_iou_map != None
#             temp_asy_iou_map = asy_iou_map[:, keep_nms_idx]
#             temp_asy_iou_map = temp_asy_iou_map > self.asy_iou_th
#
#             flag = temp_asy_iou_map * asy_iou_flag
#             org_det = det_prob_tmp[keep_nms_idx]
#
#             if flag.sum() != 0:
#                 if self.test_mode and not self.get_diffuse_gt:
#                     res_idx = keep_nms_idx[torch.sum(flag, dim=0) > 0]  # 原先的 res_idx --> 未diffuse
#                 # else:
#                 #     col_idxs = torch.sum(flag,dim=0) > 0
#                 #     flag = flag[:, col_idxs]
#                 #     org_det = org_det[col_idxs][None,:] # 1, N
#                 #     keep_nms_idx = keep_nms_idx[col_idxs]
#                 #     group_instances_idxs = group_instances_idxs[:,col_idxs]
#                 #
#                 #     assert len(keep_nms_idx) == group_instances_idxs.shape[-1]
#                 #
#                 #     res_det = flag * det_prob_tmp[:,None]
#                 #
#                 #     diffuse_idxs = res_det > org_det
#                 #     diffuse_flags = torch.sum(diffuse_idxs,dim=0) > 0
#                 #
#                 #     res_idx = []
#                 #     for idx, diffuse_flag in enumerate(diffuse_flags):
#                 #         if diffuse_flag:
#                 #             real_group_id = group_instances_idxs[:,idx]
#                 #             cam = torch.sum(proposals[real_group_id] * cls_prob_tmp[real_group_id, None, None, None], dim=0)
#                 #             cam /= (torch.max(cam) + 1e-5)
#                 #             cam = torchvision.transforms.functional.gaussian_blur(cam,11)
#                 #             refine_cam = self.PAMR_model(image, cam[None,...])
#                 #
#                 #             proposal_scores = (torch.sum(refine_cam * proposals[diffuse_idxs[:,idx]],dim=(-1,-2,-3)) - torch.sum(refine_cam * proposals[keep_nms_idx[idx]])) / \
#                 #                               (torch.sum(proposals[diffuse_idxs[:,idx]],dim=(-1,-2,-3)) - torch.sum(proposals[keep_nms_idx[idx]]))
#                 #
#                 #             max_idx = torch.argmax(proposal_scores) # 0-sum(diffuse_idxs[:, idx])
#                 #             max_idx = torch.nonzero(diffuse_idxs[:, idx])[max_idx] # 0-200 # mapping to real index
#                 #             res_idx.append(max_idx)
#                 #
#                 #         else:
#                 #             res_idx.append(keep_nms_idx[idx])
#                 #
#                 #     res_idx = torch.tensor(res_idx) # 真正的 gt
#                 #     res_idx = torch.unique(res_idx)
#
#                 else:
#                     col_idxs = torch.sum(flag,dim=0) > 0
#                     flag = flag[:, col_idxs]
#                     org_det = org_det[col_idxs][None,:] # 1, N
#                     keep_nms_idx = keep_nms_idx[col_idxs]
#                     group_instances_idxs = group_instances_idxs[:,col_idxs]
#
#                     assert len(keep_nms_idx) == group_instances_idxs.shape[-1]
#
#                     res_det = flag * det_prob_tmp[:,None]
#
#                     diffuse_idxs = res_det > org_det
#                     diffuse_flags = torch.sum(diffuse_idxs,dim=0) > 0
#
#                     res_idx = []
#                     cams = []
#                     for idx, diffuse_flag in enumerate(diffuse_flags):
#                         if diffuse_flag:
#                             real_group_id = group_instances_idxs[:,idx]
#                             cam = torch.sum(proposals[real_group_id] * cls_prob_tmp[real_group_id, None, None, None], dim=0) # 1,H,W
#                             cam /= (torch.max(cam) + 1e-5)
#                             cams.append(cam)
#
#                         else:
#                             res_idx.append(keep_nms_idx[idx])
#
#                     if len(cams) != 0:
#                         cams = torch.concat(cams,dim=0) # sum(diffuse_flags),H,W
#                         # cams = torchvision.transforms.functional.gaussian_blur(cams,11)
#                         refine_cams = self.PAMR_model(image, cams[None,...])
#
#                         i = 0
#                         for idx, diffuse_flag in enumerate(diffuse_flags):
#                             if diffuse_flag:
#                                 refine_cam = refine_cams[0, i][None,None,...]
#                                 assert refine_cam.ndim == 4
#                                 i += 1
#                                 # proposal_scores = (torch.sum(refine_cam * proposals[diffuse_idxs[:,idx]],dim=(-1,-2,-3)) - torch.sum(refine_cam * proposals[keep_nms_idx[idx]])) / \
#                                 #                   (torch.sum(proposals[diffuse_idxs[:,idx]],dim=(-1,-2,-3)) - torch.sum(proposals[keep_nms_idx[idx]]))
#
#                                 proposal_scores = torch.mean(refine_cam * proposals[diffuse_idxs[:, idx]],dim=(-1, -2, -3))
#
#                                 max_idx = torch.argmax(proposal_scores) # 0-sum(diffuse_idxs[:, idx])
#                                 max_idx = torch.nonzero(diffuse_idxs[:, idx])[max_idx] # 0-200 # mapping to real index
#                                 res_idx.append(max_idx)
#
#                     res_idx = torch.tensor(res_idx) # 真正的 gt
#                     res_idx = torch.unique(res_idx)
#
#                 is_higher_scoring_class = preds_tmp[res_idx] > gt_weights[res_idx]
#                 if is_higher_scoring_class.sum() > 0:
#                     keep_idxs = res_idx[is_higher_scoring_class]
#                     gt_labels[keep_idxs, :] = 0
#                     gt_labels[keep_idxs, c + 1] = 1
#                     gt_weights[keep_idxs] = preds_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#         # assert gt_idxs.sum() > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag
#
#     # IS score forward
#     @torch.no_grad()
#     def IS_forward(self, predict_cls, predict_det, rois, labels, iou_map=None, asy_iou_map=None, diffuse=False, image=None, proposals=None):
#         if rois.ndim == 3:
#             rois = rois.squeeze(0)
#         rois = rois[:, 1:]  # remove batch_id
#
#         if diffuse:
#             gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag = self.mist_label_diffuse_IS(predict_cls,
#                                                                                              predict_det,
#                                                                                              rois, labels, iou_map,
#                                                                                              asy_iou_map, image,
#                                                                                              proposals)
#         else:
#             if predict_det != None:
#                 preds = predict_cls * predict_det
#             else:
#                 preds = predict_cls
#
#             gt_boxes, gt_labels, gt_weights, gt_idxs = self.mist_label(preds, rois, labels, iou_map, asy_iou_map)
#
#         if gt_idxs.sum() == 0:
#             return None, None, None, None
#
#         if iou_map == None:
#             overlaps = box_iou(rois, gt_boxes)  # (proposal_num,gt_num)
#         else:
#             overlaps = iou_map[:, gt_idxs]
#
#         max_overlap_v, max_overlap_idx = torch.max(overlaps, dim=-1)
#
#         pseudo_labels = gt_labels[max_overlap_idx]
#         loss_weights = gt_weights[max_overlap_idx]
#         pseudo_iou_label = max_overlap_v
#
#         if self.test_mode:
#             bg_inds = (max_overlap_v != 1)
#             pseudo_labels[bg_inds, :] = 0
#             pseudo_labels[bg_inds, 0] = 1
#             pseudo_labels = pseudo_labels * loss_weights[:, None]
#
#             return pseudo_labels, pseudo_iou_label, loss_weights, None
#
#         else:
#             ignore_inds = max_overlap_v == 0
#             pseudo_labels[ignore_inds, :] = 0
#             loss_weights[ignore_inds] = 0
#
#             bg_inds = (max_overlap_v < self.iou_th) * ~ignore_inds
#             pseudo_labels[bg_inds, :] = 0
#             pseudo_labels[bg_inds, 0] = 1
#
#             try:
#                 # 将太大的proposal标记为bg
#                 big_proposal = ~asy_iou_flag
#                 pseudo_labels[big_proposal, :] = 0
#                 pseudo_labels[big_proposal, 0] = 1
#             except:
#                 pass
#
#             pseudo_iou_label[pseudo_iou_label > self.full_thr] = 1
#             pseudo_iou_label[pseudo_iou_label <= self.full_thr] = 0
#
#             # pseudo_iou_label = (pseudo_iou_label - self.iou_th) / (1 - self.iou_th)
#             # pseudo_iou_label = (pseudo_iou_label - 0.5) / (1 - 0.5)
#
#             group_assign = max_overlap_idx + 1  # 将 index 从 1 开始
#             group_assign[bg_inds] = -1
#             group_assign[ignore_inds] = -2
#             # 所以最后的范围是 [-2, -1, , 1, 2, 3...]
#
#             # transform pseudo_labels --> N * 21
#             # 将 group_assign 变为 和 初始伪标签 mat 相同的形态
#             group_assign = group_assign[:, None] * pseudo_labels
#
#             return pseudo_labels, pseudo_iou_label, loss_weights, group_assign


# # 对 gt 进行 sample
# class mist_layer_with_sample_gt(nn.Module):
#     def __init__(self, portion=0.1, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85):
#         super(mist_layer_with_sample_gt, self).__init__()
#         self.portion = portion
#         self.full_thr = full_thr
#         self.iou_th = iou_thr
#         self.asy_iou_th = asy_iou_th
#
#         print("mist_layer_with_sample_gt--> portion:{}, full_thr: {}, iou_thr: {}".format(portion,full_thr,iou_thr))
#
#     # instance_list -> [{},{}...]
#     # {} -> {score: float, mask_id: int}
#     def instance_nms(self, instance_list, iou_map):
#         instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)
#
#         selected_instances_id = []
#         while len(instance_list) > 0:
#             src_instance = instance_list.pop(0)
#             selected_instances_id.append(src_instance["mask_id"])
#
#             src_mask_id = src_instance["mask_id"]
#
#             def iou_filter(dst_instance):
#                 dst_mask_id = dst_instance["mask_id"]
#
#                 iou = iou_map[src_mask_id][dst_mask_id]
#                 if iou < self.iou_th:
#                     return True
#                 else:
#                     return False
#
#             instance_list = list(filter(iou_filter, instance_list))
#
#         return selected_instances_id
#
#     # instance_list -> [{},{}...]
#     # {} -> {score: float, mask_id: int}
#     def instance_asy_nms(self, instance_list, asy_iou_map):
#         instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)
#
#         selected_instances_id = []
#         while len(instance_list) > 0:
#             src_instance = instance_list.pop(0)
#             selected_instances_id.append(src_instance["mask_id"])
#
#             src_mask_id = src_instance["mask_id"]
#
#             def iou_filter(dst_instance):
#                 dst_mask_id = dst_instance["mask_id"]
#
#                 iou = asy_iou_map[src_mask_id][dst_mask_id]
#                 if iou < self.asy_iou_th:
#                     return True
#                 else:
#                     return False
#
#             instance_list = list(filter(iou_filter, instance_list))
#
#         return selected_instances_id
#
#     @torch.no_grad()
#     def mist_label(self, preds, rois, label, iou_map=None, asy_iou_map=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         preds = (preds if preds.shape[-1] == label.shape[-1] else preds[:, 1:]).clone()  # remove background class if present
#         keep_count = int(np.ceil(self.portion * preds.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((preds.shape[0], label.shape[-1] + 1), dtype=preds.dtype, device=preds.device)
#         gt_weights = -torch.ones((preds.shape[0],), dtype=preds.dtype, device=preds.device)
#
#         for c in klasses:
#             cls_prob_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=preds.device)
#
#             # box nms
#             else:
#                 print("iou_map == None")
#                 keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             is_higher_scoring_class = cls_prob_tmp[keep_nms_idx] > gt_weights[keep_nms_idx]
#             keep_idxs = keep_nms_idx[is_higher_scoring_class]
#             gt_labels[keep_idxs, :] = 0
#             gt_labels[keep_idxs, c + 1] = 1
#             gt_weights[keep_idxs] = cls_prob_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs, klasses
#
#     @torch.no_grad()
#     def mist_label_diffuse(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
#         predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present
#
#         preds = predict_cls * predict_det
#
#         keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
#         gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)
#
#         # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
#         asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]
#
#         for c in klasses:
#             cls_prob_tmp = predict_cls[:, c]
#             if predict_det.shape[-1] == label.shape[-1]:
#                 det_prob_tmp = predict_det[:, c]
#             elif predict_det.shape[-1] == 1:
#                 det_prob_tmp = predict_det[:, 0]
#             else:
#                 raise AssertionError
#
#             preds_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # cal iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)
#
#             # box nms
#             else:
#                 print("iou_map == None")
#                 keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             assert asy_iou_map != None
#             temp_asy_iou_map = asy_iou_map[:, keep_nms_idx]
#             temp_asy_iou_map = temp_asy_iou_map > self.asy_iou_th
#
#             flag = temp_asy_iou_map * asy_iou_flag
#             if flag.sum() != 0:
#                 flag = flag[:, torch.sum(flag,dim=0) > 0]
#                 res_det = flag * det_prob_tmp[:,None]
#                 res_idx = torch.argmax(res_det, dim=0) # 真正的 gt
#                 res_idx = torch.unique(res_idx)
#
#                 is_higher_scoring_class = preds_tmp[res_idx] > gt_weights[res_idx]
#                 if is_higher_scoring_class.sum() > 0:
#                     keep_idxs = res_idx[is_higher_scoring_class]
#                     gt_labels[keep_idxs, :] = 0
#                     gt_labels[keep_idxs, c + 1] = 1
#                     gt_weights[keep_idxs] = preds_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#         # assert gt_idxs.sum() > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag, klasses
#
#     # == 0, ignore
#     # != 0 and < 0.25, bg
#     # >= 0.25, fg
#     # < 0.5, complete
#     # >= 0.5, not complete
#     @torch.no_grad()
#     def forward(self, predict_cls, predict_det, rois, labels, iou_map=None, asy_iou_map=None, diffuse = False):
#         if rois.ndim == 3:
#             rois = rois.squeeze(0)
#         rois = rois[:,1:] # remove batch_id
#
#         if diffuse:
#             gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag, klasses = self.mist_label_diffuse(predict_cls, predict_det, rois, labels, iou_map, asy_iou_map)
#         else:
#             if predict_det!= None:
#                 preds = predict_cls * predict_det
#             else:
#                 preds = predict_cls
#
#             gt_boxes, gt_labels, gt_weights, gt_idxs, klasses = self.mist_label(preds,rois,labels,iou_map,asy_iou_map)
#
#         if gt_idxs.sum() == 0:
#             return None, None, None, None
#
#         if iou_map == None:
#             overlaps = box_iou(rois, gt_boxes) # (proposal_num,gt_num)
#         else:
#             overlaps = iou_map[:, gt_idxs]
#
#         # 对 gt 进行 sample
#         # 消除面积对 proposal 的 bias
#
#         # False --> 清除位
#         # True  --> 保持位
#         inds = torch.ones_like(gt_labels[:, 0], device=gt_labels.device)
#
#         for c in klasses:
#             class_idx = torch.nonzero(gt_labels[:, c + 1] == 1).flatten().cpu().numpy()
#             if len(class_idx) == 0:
#                 continue
#
#             prob = gt_weights[class_idx].cpu().numpy()
#             sampled_class_idx = np.random.choice(class_idx, size=len(class_idx), replace=True,
#                                                p=prob / prob.sum())
#             sampled_class_idx = np.unique(sampled_class_idx)
#
#             inds[class_idx] = 0
#             inds[sampled_class_idx] = 1
#
#         inds = inds == 1
#         gt_weights = gt_weights[inds]
#         gt_labels = gt_labels[inds, :]
#         gt_boxes = gt_boxes[inds, :]
#         overlaps = overlaps[:,inds]
#
#         # sample done
#         ################
#
#         max_overlap_v, max_overlap_idx = torch.max(overlaps,dim=-1)
#
#         pseudo_labels = gt_labels[max_overlap_idx]
#         loss_weights = gt_weights[max_overlap_idx]
#         pseudo_iou_label = max_overlap_v
#
#         ignore_inds = max_overlap_v == 0
#         pseudo_labels[ignore_inds, :] = 0
#         loss_weights[ignore_inds] = 0
#
#         bg_inds = (max_overlap_v < self.iou_th) * ~ignore_inds
#         pseudo_labels[bg_inds,:] = 0
#         pseudo_labels[bg_inds,0] = 1
#
#         try:
#             # 将太大的proposal标记为bg
#             big_proposal = ~asy_iou_flag
#             pseudo_labels[big_proposal, :] = 0
#             pseudo_labels[big_proposal, 0] = 1
#         except:
#             pass
#
#         pseudo_iou_label[pseudo_iou_label > self.full_thr] = 1
#         pseudo_iou_label[pseudo_iou_label <= self.full_thr] = 0
#
#         # pseudo_iou_label = (pseudo_iou_label - self.iou_th) / (1 - self.iou_th)
#         # pseudo_iou_label = (pseudo_iou_label - 0.5) / (1 - 0.5)
#
#         group_assign = max_overlap_idx + 1 # 将 index 从 1 开始
#         group_assign[bg_inds] = -1
#         group_assign[ignore_inds] = -2
#         # 所以最后的范围是 [-2, -1, , 1, 2, 3...]
#
#         # transform pseudo_labels --> N * 21
#         # 将 group_assign 变为 和 初始伪标签 mat 相同的形态
#         group_assign = group_assign[:,None] * pseudo_labels
#
#         return pseudo_labels, pseudo_iou_label, loss_weights, group_assign

# # 使用 bu 信息对 proposal 进行选择
# class mist_layer_with_bu(nn.Module):
#     def __init__(self, portion=0.1, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85):
#         super(mist_layer_with_bu, self).__init__()
#         self.portion = portion
#         self.full_thr = full_thr
#         self.iou_th = iou_thr
#         self.asy_iou_th = asy_iou_th
#
#         print("mist_layer_with_bu--> portion:{}, full_thr: {}, iou_thr: {}".format(portion,full_thr,iou_thr))
#
#     # instance_list -> [{},{}...]
#     # {} -> {score: float, mask_id: int}
#     def instance_nms(self, instance_list, iou_map):
#         instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)
#
#         selected_instances_id = []
#         while len(instance_list) > 0:
#             src_instance = instance_list.pop(0)
#             selected_instances_id.append(src_instance["mask_id"])
#
#             src_mask_id = src_instance["mask_id"]
#
#             def iou_filter(dst_instance):
#                 dst_mask_id = dst_instance["mask_id"]
#
#                 iou = iou_map[src_mask_id][dst_mask_id]
#                 if iou < self.iou_th:
#                     return True
#                 else:
#                     return False
#
#             instance_list = list(filter(iou_filter, instance_list))
#
#         return selected_instances_id
#
#     # instance_list -> [{},{}...]
#     # {} -> {score: float, mask_id: int}
#     def instance_asy_nms(self, instance_list, asy_iou_map):
#         instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)
#
#         selected_instances_id = []
#         while len(instance_list) > 0:
#             src_instance = instance_list.pop(0)
#             selected_instances_id.append(src_instance["mask_id"])
#
#             src_mask_id = src_instance["mask_id"]
#
#             def iou_filter(dst_instance):
#                 dst_mask_id = dst_instance["mask_id"]
#
#                 iou = asy_iou_map[src_mask_id][dst_mask_id]
#                 if iou < self.asy_iou_th:
#                     return True
#                 else:
#                     return False
#
#             instance_list = list(filter(iou_filter, instance_list))
#
#         return selected_instances_id
#
#     @torch.no_grad()
#     def mist_label(self, preds, rois, label, iou_map=None, asy_iou_map=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         preds = (preds if preds.shape[-1] == label.shape[-1] else preds[:, 1:]).clone()  # remove background class if present
#         keep_count = int(np.ceil(self.portion * preds.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((preds.shape[0], label.shape[-1] + 1), dtype=preds.dtype, device=preds.device)
#         gt_weights = -torch.ones((preds.shape[0],), dtype=preds.dtype, device=preds.device)
#
#         for c in klasses:
#             cls_prob_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=preds.device)
#
#             # box nms
#             else:
#                 print("iou_map == None")
#                 keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             is_higher_scoring_class = cls_prob_tmp[keep_nms_idx] > gt_weights[keep_nms_idx]
#             keep_idxs = keep_nms_idx[is_higher_scoring_class]
#             gt_labels[keep_idxs, :] = 0
#             gt_labels[keep_idxs, c + 1] = 1
#             gt_weights[keep_idxs] = cls_prob_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs
#
#     @torch.no_grad()
#     def mist_label_diffuse(self, predict_cls, predict_det, rois, label, iou_map, asy_iou_map, bu_cues):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
#         predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present
#
#         preds = predict_cls * predict_det
#
#         keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
#         gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)
#
#         # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
#         asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]
#
#         for c in klasses:
#             cls_prob_tmp = predict_cls[:, c]
#             if predict_det.shape[-1] == label.shape[-1]:
#                 det_prob_tmp = predict_det[:, c]
#             elif predict_det.shape[-1] == 1:
#                 det_prob_tmp = predict_det[:, 0]
#             else:
#                 raise AssertionError
#
#             preds_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # cal iou nms
#             temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#             instance_list = []
#             for i, prob in enumerate(keep_cls_prob):
#                 instance = dict()
#
#                 instance["score"] = prob
#                 instance["mask_id"] = i
#                 instance_list.append(instance)
#
#             keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#             keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             assert asy_iou_map != None
#             temp_asy_iou_map = asy_iou_map[:, keep_nms_idx]
#             temp_asy_iou_map = temp_asy_iou_map > self.asy_iou_th
#
#             flag = temp_asy_iou_map * asy_iou_flag
#             org_det = det_prob_tmp[keep_nms_idx]
#
#             if flag.sum() != 0:
#                 col_idxs = torch.sum(flag,dim=0) > 0
#                 flag = flag[:, col_idxs]
#                 org_det = org_det[col_idxs][None,:] # 1, N
#
#                 res_det = flag * det_prob_tmp[:,None]
#
#                 diffuse_idxs = res_det > org_det
#                 diffuse_flag = torch.sum(diffuse_idxs,dim=0,keepdim=True) > 0
#                 res_idx = torch.argmax(
#                                         (diffuse_flag * diffuse_idxs * bu_cues[:,None] + (~diffuse_flag) * res_det),
#                                         dim=0) # 真正的 gt
#
#                 res_idx = torch.unique(res_idx)
#
#                 is_higher_scoring_class = preds_tmp[res_idx] > gt_weights[res_idx]
#                 if is_higher_scoring_class.sum() > 0:
#                     keep_idxs = res_idx[is_higher_scoring_class]
#                     gt_labels[keep_idxs, :] = 0
#                     gt_labels[keep_idxs, c + 1] = 1
#                     gt_weights[keep_idxs] = preds_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag
#
#     # == 0, ignore
#     # != 0 and < 0.25, bg
#     # >= 0.25, fg
#     # < 0.5, complete
#     # >= 0.5, not complete
#     @torch.no_grad()
#     def forward(self, predict_cls, predict_det, rois, labels, iou_map, asy_iou_map,bu_cues, diffuse=False):
#         if rois.ndim == 3:
#             rois = rois.squeeze(0)
#         rois = rois[:, 1:]  # remove batch_id
#
#         if diffuse:
#             gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag = self.mist_label_diffuse(predict_cls, predict_det,
#                                                                                              rois, labels, iou_map,
#                                                                                              asy_iou_map,bu_cues)
#         else:
#             if predict_det != None:
#                 preds = predict_cls * predict_det
#             else:
#                 preds = predict_cls
#
#             gt_boxes, gt_labels, gt_weights, gt_idxs = self.mist_label(preds, rois, labels, iou_map, asy_iou_map)
#
#         if gt_idxs.sum() == 0:
#             return None, None, None, None
#
#         if iou_map == None:
#             overlaps = box_iou(rois, gt_boxes)  # (proposal_num,gt_num)
#         else:
#             overlaps = iou_map[:, gt_idxs]
#
#         max_overlap_v, max_overlap_idx = torch.max(overlaps, dim=-1)
#
#         pseudo_labels = gt_labels[max_overlap_idx]
#         loss_weights = gt_weights[max_overlap_idx]
#         pseudo_iou_label = max_overlap_v
#
#         ignore_inds = max_overlap_v == 0
#         pseudo_labels[ignore_inds, :] = 0
#         loss_weights[ignore_inds] = 0
#
#         bg_inds = (max_overlap_v < self.iou_th) * ~ignore_inds
#         pseudo_labels[bg_inds, :] = 0
#         pseudo_labels[bg_inds, 0] = 1
#
#         try:
#             # 将太大的proposal标记为bg
#             big_proposal = ~asy_iou_flag
#             pseudo_labels[big_proposal, :] = 0
#             pseudo_labels[big_proposal, 0] = 1
#         except:
#             pass
#
#         pseudo_iou_label[pseudo_iou_label > self.full_thr] = 1
#         pseudo_iou_label[pseudo_iou_label <= self.full_thr] = 0
#
#         group_assign = max_overlap_idx + 1  # 将 index 从 1 开始
#         group_assign[bg_inds] = -1
#         group_assign[ignore_inds] = -2
#         # 所以最后的范围是 [-2, -1, , 1, 2, 3...]
#
#         # transform pseudo_labels --> N * 21
#         # 将 group_assign 变为 和 初始伪标签 mat 相同的形态
#         group_assign = group_assign[:, None] * pseudo_labels
#
#         return pseudo_labels, pseudo_iou_label, loss_weights, group_assign
#
# # 将 gt 内部的的全部 proposal 对设置为对应的类别
# class mist_layer_with_inner_cls(nn.Module):
#     def __init__(self, portion=0.1, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85):
#         super(mist_layer_with_inner_cls, self).__init__()
#         self.portion = portion
#         self.full_thr = full_thr
#         self.iou_th = iou_thr
#         self.asy_iou_th = asy_iou_th
#
#         print("mist_layer_with_inner_cls--> portion:{}, full_thr: {}, iou_thr: {}".format(portion,full_thr,iou_thr))
#
#     # instance_list -> [{},{}...]
#     # {} -> {score: float, mask_id: int}
#     def instance_nms(self, instance_list, iou_map):
#         instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)
#
#         selected_instances_id = []
#         while len(instance_list) > 0:
#             src_instance = instance_list.pop(0)
#             selected_instances_id.append(src_instance["mask_id"])
#
#             src_mask_id = src_instance["mask_id"]
#
#             def iou_filter(dst_instance):
#                 dst_mask_id = dst_instance["mask_id"]
#
#                 iou = iou_map[src_mask_id][dst_mask_id]
#                 if iou < self.iou_th:
#                     return True
#                 else:
#                     return False
#
#             instance_list = list(filter(iou_filter, instance_list))
#
#         return selected_instances_id
#
#     # instance_list -> [{},{}...]
#     # {} -> {score: float, mask_id: int}
#     def instance_asy_nms(self, instance_list, asy_iou_map):
#         instance_list = sorted(instance_list, key=lambda x: x["score"], reverse=True)
#
#         selected_instances_id = []
#         while len(instance_list) > 0:
#             src_instance = instance_list.pop(0)
#             selected_instances_id.append(src_instance["mask_id"])
#
#             src_mask_id = src_instance["mask_id"]
#
#             def iou_filter(dst_instance):
#                 dst_mask_id = dst_instance["mask_id"]
#
#                 iou = asy_iou_map[src_mask_id][dst_mask_id]
#                 if iou < self.asy_iou_th:
#                     return True
#                 else:
#                     return False
#
#             instance_list = list(filter(iou_filter, instance_list))
#
#         return selected_instances_id
#
#     @torch.no_grad()
#     def mist_label(self, preds, rois, label, iou_map=None, asy_iou_map=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         preds = (preds if preds.shape[-1] == label.shape[-1] else preds[:, 1:]).clone()  # remove background class if present
#         keep_count = int(np.ceil(self.portion * preds.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((preds.shape[0], label.shape[-1] + 1), dtype=preds.dtype, device=preds.device)
#         gt_weights = -torch.ones((preds.shape[0],), dtype=preds.dtype, device=preds.device)
#
#         for c in klasses:
#             cls_prob_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=preds.device)
#
#             # box nms
#             else:
#                 print("iou_map == None")
#                 keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             is_higher_scoring_class = cls_prob_tmp[keep_nms_idx] > gt_weights[keep_nms_idx]
#             keep_idxs = keep_nms_idx[is_higher_scoring_class]
#             gt_labels[keep_idxs, :] = 0
#             gt_labels[keep_idxs, c + 1] = 1
#             gt_weights[keep_idxs] = cls_prob_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs
#
#     @torch.no_grad()
#     def mist_label_diffuse(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
#         predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present
#
#         preds = predict_cls * predict_det
#
#         keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
#         gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)
#
#         # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
#         asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]
#
#         for c in klasses:
#             cls_prob_tmp = predict_cls[:, c]
#             if predict_det.shape[-1] == label.shape[-1]:
#                 det_prob_tmp = predict_det[:, c]
#             elif predict_det.shape[-1] == 1:
#                 det_prob_tmp = predict_det[:, 0]
#             else:
#                 raise AssertionError
#
#             preds_tmp = preds[:, c]
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # cal iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)
#
#             # box nms
#             else:
#                 print("iou_map == None")
#                 keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             assert asy_iou_map != None
#             temp_asy_iou_map = asy_iou_map[:, keep_nms_idx]
#             temp_asy_iou_map = temp_asy_iou_map > self.asy_iou_th
#
#             flag = temp_asy_iou_map * asy_iou_flag
#             if flag.sum() != 0:
#                 flag = flag[:, torch.sum(flag,dim=0) > 0]
#                 res_det = flag * det_prob_tmp[:,None]
#                 res_idx = torch.argmax(res_det, dim=0) # 真正的 gt
#                 res_idx = torch.unique(res_idx)
#
#                 is_higher_scoring_class = preds_tmp[res_idx] > gt_weights[res_idx]
#                 if is_higher_scoring_class.sum() > 0:
#                     keep_idxs = res_idx[is_higher_scoring_class]
#                     gt_labels[keep_idxs, :] = 0
#                     gt_labels[keep_idxs, c + 1] = 1
#                     gt_weights[keep_idxs] = preds_tmp[keep_idxs]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#         # assert gt_idxs.sum() > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag
#
#     # loss weight 使用扩散以后的iou score，和原本的 cls score构成
#     @torch.no_grad()
#     def mist_label_diffuse_reweight(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None):
#         if label.dim() != 1:
#             label = label.squeeze()
#
#         assert label.dim() == 1
#         assert label.shape[-1] == 20 or label.shape[-1] == 80
#
#         # bg remove
#         predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
#         predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present
#
#         keep_count = int(np.ceil(self.portion * predict_cls.shape[0]))
#         klasses = label.nonzero(as_tuple=True)[0]
#         # one hot label
#         gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
#         gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)
#
#         # 将面积过大的 proposal 删除，比如说proposal占一整张图片的情况
#         asy_iou_flag = torch.sum(asy_iou_map > self.asy_iou_th, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]
#
#         for c in klasses:
#             cls_prob_tmp = predict_cls[:, c]
#             if predict_det.shape[-1] == label.shape[-1]:
#                 det_prob_tmp = predict_det[:, c]
#             elif predict_det.shape[-1] == 1:
#                 det_prob_tmp = predict_det[:, 0]
#             else:
#                 raise AssertionError
#
#             keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]  # top p percent of proposals
#
#             keep_rois = rois[keep_sort_idx]
#             keep_cls_prob = cls_prob_tmp[keep_sort_idx]
#
#             # cal iou nms
#             if iou_map != None:
#                 temp_iou_map = iou_map[keep_sort_idx][:, keep_sort_idx]
#
#                 instance_list = []
#                 for i, prob in enumerate(keep_cls_prob):
#                     instance = dict()
#
#                     instance["score"] = prob
#                     instance["mask_id"] = i
#                     instance_list.append(instance)
#
#                 keep_nms_idx = self.instance_nms(instance_list, temp_iou_map)
#                 keep_nms_idx = torch.tensor(keep_nms_idx,device=predict_cls.device)
#
#             # box nms
#             else:
#                 print("asy_iou_map == None")
#                 keep_nms_idx = nms(keep_rois, keep_cls_prob, self.iou_th)
#
#             keep_nms_idx = keep_sort_idx[keep_nms_idx]  # mapping index to org index
#
#             assert asy_iou_map != None
#             temp_asy_iou_map = asy_iou_map[:, keep_nms_idx]
#             temp_asy_iou_map = temp_asy_iou_map > self.asy_iou_th
#
#             flag = temp_asy_iou_map * asy_iou_flag
#             if flag.sum() != 0:
#                 avi_idx = torch.sum(flag,dim=0) > 0
#                 flag = flag[:, avi_idx]
#                 keep_nms_idx = keep_nms_idx[avi_idx]
#                 res_det = flag * det_prob_tmp[:,None]
#                 res_idx = torch.argmax(res_det, dim=0) # 真正的 gt idx
#
#                 new_weight = det_prob_tmp[res_idx] * cls_prob_tmp[keep_nms_idx]
#
#                 is_higher_scoring_class = new_weight > gt_weights[res_idx]
#                 if is_higher_scoring_class.sum() > 0:
#                     keep_idxs = res_idx[is_higher_scoring_class]
#                     gt_labels[keep_idxs, :] = 0
#                     gt_labels[keep_idxs, c + 1] = 1
#                     gt_weights[keep_idxs] = new_weight[is_higher_scoring_class]
#
#         gt_idxs = torch.sum(gt_labels, dim=-1) > 0
#         # assert gt_idxs.sum() > 0
#
#         gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]
#
#         return gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag
#
#     # == 0, ignore
#     # != 0 and < 0.25, bg
#     # >= 0.25, fg
#     # < 0.5, complete
#     # >= 0.5, not complete
#     @torch.no_grad()
#     def forward(self, predict_cls, predict_det, rois, labels, iou_map=None, asy_iou_map=None, diffuse = False):
#         if rois.ndim == 3:
#             rois = rois.squeeze(0)
#         rois = rois[:,1:] # remove batch_id
#
#         if diffuse:
#             gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag = self.mist_label_diffuse(predict_cls, predict_det, rois, labels, iou_map, asy_iou_map)
#             # gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag = self.mist_label_diffuse_reweight(predict_cls, predict_det, rois, labels, iou_map, asy_iou_map)
#         else:
#             if predict_det!= None:
#                 preds = predict_cls * predict_det
#             else:
#                 preds = predict_cls
#
#             gt_boxes, gt_labels, gt_weights, gt_idxs = self.mist_label(preds,rois,labels,iou_map,asy_iou_map)
#
#         if gt_idxs.sum() == 0:
#             return None, None, None, None
#
#         if iou_map == None:
#             overlaps = box_iou(rois, gt_boxes) # (proposal_num,gt_num)
#         else:
#             overlaps = iou_map[:, gt_idxs]
#
#         max_overlap_v, max_overlap_idx = torch.max(overlaps,dim=-1)
#
#         pseudo_labels = gt_labels[max_overlap_idx]
#         loss_weights = gt_weights[max_overlap_idx]
#         pseudo_iou_label = max_overlap_v
#
#         ignore_inds = max_overlap_v == 0
#         pseudo_labels[ignore_inds, :] = 0
#         loss_weights[ignore_inds] = 0
#
#         # bg_inds = (max_overlap_v < self.iou_th) * ~ignore_inds
#         # pseudo_labels[bg_inds,:] = 0
#         # pseudo_labels[bg_inds,0] = 1
#
#         gt_asy_v = asy_iou_map[gt_idxs,:]
#         max_asy_v = torch.gather(gt_asy_v, dim=0, index=max_overlap_idx[None,...]).flatten()
#
#         # 将包含在 gt 里面的 proposal 从背景中 设置为对应的类别
#         bg_inds = (max_overlap_v < self.iou_th) * (max_asy_v <= self.asy_iou_th) * ~ignore_inds
#         pseudo_labels[bg_inds,:] = 0
#         pseudo_labels[bg_inds,0] = 1
#
#         try:
#             # 将太大的proposal标记为bg
#             big_proposal = ~asy_iou_flag
#             pseudo_labels[big_proposal, :] = 0
#             pseudo_labels[big_proposal, 0] = 1
#         except:
#             pass
#
#         pseudo_iou_label[pseudo_iou_label > self.full_thr] = 1
#         pseudo_iou_label[pseudo_iou_label <= self.full_thr] = 0
#
#         # pseudo_iou_label = (pseudo_iou_label - self.iou_th) / (1 - self.iou_th)
#         # pseudo_iou_label = (pseudo_iou_label - 0.5) / (1 - 0.5)
#
#         group_assign = max_overlap_idx + 1 # 将 index 从 1 开始
#         group_assign[bg_inds] = -1
#         group_assign[ignore_inds] = -2
#         # 所以最后的范围是 [-2, -1, , 1, 2, 3...]
#
#         # transform pseudo_labels --> N * 21
#         # 将 group_assign 变为 和 初始伪标签 mat 相同的形态
#         group_assign = group_assign[:,None] * pseudo_labels
#
#         return pseudo_labels, pseudo_iou_label, loss_weights, group_assign

def mil_losses(cls_score, labels,loss_weight=None):
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)  # min-max  [1e-6,1 - 1e-6]
    labels = labels.clamp(0, 1)
    loss = -labels * torch.log(cls_score) - (1 - labels) * torch.log(1 - cls_score)
    if loss_weight != None:
        loss = loss * loss_weight

    return loss.mean()

def peak_cluster_losses(refine, gt_assignment, loss_function, labels=None, peak_score=None):
    assert len(refine) == len(gt_assignment)
    batch_size, channels = refine.size()
    refine = refine.clamp(1e-9, 1 - 1e-9)

    if loss_function == 'two_BCELoss':
        label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
        label_tmp[:, 1:] = labels
        aggregation = refine.max(0)[0]
        loss2 = mil_losses(aggregation, label_tmp) # bag loss
        ind = (gt_assignment != 0).sum(1) != 0
        refine_tmp = refine[ind, :]
        gt_tmp = (gt_assignment[ind, :] != 0).float()
        class_num = labels.shape[1] + 1
        # score = peak_score[ind, :]
        if len(gt_tmp) != 0:
            loss1 = mil_losses(refine_tmp, gt_tmp) # instance loss
            # loss1 = mil_losses(refine_tmp, gt_tmp, score)
        else:
            loss1 = torch.tensor(0.).to(device=loss2.device)
        return loss1 * class_num, loss2
        # return loss1 * label_tmp.sum() , loss2

    elif loss_function == 'two_BCELoss_w':
        label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
        label_tmp[:, 1:] = labels
        aggregation = refine.max(0)[0]
        loss2 = mil_losses(aggregation, label_tmp)
        ind = (gt_assignment != 0).sum(1) != 0
        refine_tmp = refine[ind, :]
        gt_tmp = (gt_assignment[ind, :] != 0).float()

        loss_weight = torch.sum(refine_tmp * gt_tmp,dim=1,keepdim=True).detach()
        if len(gt_tmp) != 0:
            loss1 = mil_losses(refine_tmp, gt_tmp, loss_weight)
        else:
            loss1 = torch.tensor(0.).cuda(device=loss2.device)
        return 4 * loss1, loss2

    elif loss_function == 'two_BCELoss_fake':
        label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
        label_tmp[:, 1:] = labels
        aggregation = refine.max(0)[0]
        loss2 = mil_losses(aggregation, label_tmp)
        ind = (gt_assignment != 0).sum(1) != 0
        refine_tmp = refine[ind, :]
        gt_tmp = (gt_assignment[ind, :] != 0).float()

        if len(gt_tmp) != 0:
            loss1 = mil_losses(refine_tmp, gt_tmp)
        else:
            loss1 = torch.tensor(0.).cuda(device=loss2.device)
        return 4 * loss1, loss2

    elif loss_function == 'graph_two_Loss':
        # aggregation = refine.max(0)[0][1:]
        # aggregation.reshape(labels.shape[0], labels.shape[1])
        #
        # loss2 = mil_losses(aggregation, labels)

        # 已经测过了
        label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
        label_tmp[:, 1:] = labels
        aggregation = refine.max(0)[0]
        loss2 = mil_losses(aggregation, label_tmp)

        loss1 = torch.tensor(0.).cuda(device=loss2.device)

        bg_ind = np.setdiff1d(gt_assignment[:,0].cpu().numpy(), [0])
        if len(bg_ind) == 0:
            # no bg
            bg_ind = 10000
        else:
            assert len(bg_ind) == 1
            bg_ind = bg_ind[0]
        # bg_ind = gt_assignment.max().item()
        fg_bg_num = 1e-6
        for cluster_ind in gt_assignment.unique():
            if cluster_ind.item() != 0 and cluster_ind.item() != bg_ind:
                TFmat = (gt_assignment == cluster_ind)
                refine_tmp = refine[TFmat.sum(1) != 0, :]
                # if cfg.SCORE:
                #     score = peak_score[TFmat.sum(1) != 0, :]
                #     col_ind = (TFmat.sum(0) != 0).float()
                #     refine_tmp_vector = refine_tmp.max(0)[0]
                #     # print(refine_tmp_vector, col_ind)
                #     fg_bg_num += len(refine_tmp)
                #     loss1 += score.mean().item() * len(refine_tmp) * mil_losses(refine_tmp_vector, col_ind)
                #     # loss1 += score.mean().item() * mil_losses(refine_tmp_vector, col_ind)
                # else:
                #     col_ind = (TFmat.sum(0) != 0).float()
                #     refine_tmp_vector = refine_tmp.max(0)[0]
                #     # print(refine_tmp_vector, col_ind)
                #     fg_bg_num += len(refine_tmp)
                #     loss1 += len(refine_tmp) * mil_losses(refine_tmp_vector, col_ind)
                #     # loss1 += mil_losses(refine_tmp_vector, col_ind)

                col_ind = (TFmat.sum(0) != 0).float()
                refine_tmp_vector = refine_tmp.max(0)[0]
                # print(refine_tmp_vector, col_ind)
                fg_bg_num += refine_tmp.shape[0]
                loss1 += refine_tmp.shape[0] * mil_losses(refine_tmp_vector, col_ind)

            elif cluster_ind.item() == bg_ind:
                TFmat = (gt_assignment == cluster_ind)
                refine_tmp = refine[TFmat.sum(1) != 0, :]
                gt_tmp = (gt_assignment[TFmat.sum(1) != 0, :] != 0).float()
                fg_bg_num += refine_tmp.shape[0]
                loss1 += refine_tmp.shape[0] * mil_losses(refine_tmp, gt_tmp)

        loss1 = loss1 / fg_bg_num
        return 4 * loss1, loss2

        # loss1 = loss1 / fg_bg_num
        # return loss1, loss2 * 0.5

    elif loss_function == 'graph_two_Loss_mean':
        # 已经测过了
        label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
        label_tmp[:, 1:] = labels
        aggregation = refine.max(0)[0]
        loss2 = mil_losses(aggregation, label_tmp)

        loss1 = torch.tensor(0.).cuda(device=loss2.device)

        bg_ind = np.setdiff1d(gt_assignment[:,0].cpu().numpy(), [0])
        if len(bg_ind) == 0:
            # no bg
            bg_ind = 10000
        else:
            assert len(bg_ind) == 1
            bg_ind = bg_ind[0]
        # bg_ind = gt_assignment.max().item()
        fg_bg_num = 1e-6
        for cluster_ind in gt_assignment.unique():
            if cluster_ind.item() != 0 and cluster_ind.item() != bg_ind:
                TFmat = (gt_assignment == cluster_ind)
                refine_tmp = refine[TFmat.sum(1) != 0, :]
                col_ind = (TFmat.sum(0) != 0).float()
                refine_tmp_vector = refine_tmp.mean(0)
                fg_bg_num += refine_tmp.shape[0]
                loss1 += refine_tmp.shape[0] * mil_losses(refine_tmp_vector, col_ind)

            elif cluster_ind.item() == bg_ind:
                TFmat = (gt_assignment == cluster_ind)
                refine_tmp = refine[TFmat.sum(1) != 0, :]
                gt_tmp = (gt_assignment[TFmat.sum(1) != 0, :] != 0).float()
                fg_bg_num += refine_tmp.shape[0]
                loss1 += refine_tmp.shape[0] * mil_losses(refine_tmp, gt_tmp)

        loss1 = loss1 / fg_bg_num
        return 4 * loss1, loss2

    elif loss_function == 'jr_loss':
        ind = (gt_assignment != 0).sum(1) != 0
        refine_tmp = refine[ind, :]
        gt_tmp = (gt_assignment[ind, :] != 0).float()
        class_num = labels.shape[1] + 1
        if len(gt_tmp) != 0:
            loss1 = mil_losses(refine_tmp, gt_tmp)  # instance loss
        else:
            loss1 = torch.tensor(0., requires_grad=True, device=refine.device)


        loss2 = torch.tensor(0.,dtype=torch.float32,device=refine.device, requires_grad=True)
        bg_ind = np.setdiff1d(gt_assignment[:, 0].cpu().numpy(), [0])
        if len(bg_ind) == 0:
            # no bg
            bg_ind = 10000
        else:
            assert len(bg_ind) == 1
            bg_ind = bg_ind[0]
        # bg_ind = gt_assignment.max().item()
        fg_bg_num = 1e-6
        for cluster_ind in gt_assignment.unique():
            if cluster_ind.item() != 0 and cluster_ind.item() != bg_ind:
                TFmat = (gt_assignment == cluster_ind)
                refine_tmp = refine[TFmat.sum(1) != 0, :]

                col_ind = (TFmat.sum(0) != 0).float()
                refine_tmp_vector = refine_tmp.max(0)[0]
                fg_bg_num += refine_tmp.shape[0]
                loss2 = loss2 + refine_tmp.shape[0] * mil_losses(refine_tmp_vector, col_ind)

            elif cluster_ind.item() == bg_ind:
                TFmat = (gt_assignment == cluster_ind)
                refine_tmp = refine[TFmat.sum(1) != 0, :]
                gt_tmp = (gt_assignment[TFmat.sum(1) != 0, :] != 0).float()
                fg_bg_num += refine_tmp.shape[0]
                loss2 = loss2 + refine_tmp.shape[0] * mil_losses(refine_tmp, gt_tmp)

        loss2 = loss2 / fg_bg_num

        return 4 * loss1, loss2

    elif loss_function == "max_mil":
        label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
        label_tmp[:, 1:] = labels
        aggregation = refine.max(0)[0]
        loss2 = mil_losses(aggregation, label_tmp)  # bag loss

        return torch.tensor(0.,device=loss2.device,requires_grad=True), loss2

    else:
        raise ValueError('loss_function should be GroupCELoss_v1, GroupCELoss, MSELoss or CELoss')

# ban
class cls_iou_loss(nn.Module):
    def __init__(self):
        super(cls_iou_loss, self).__init__()

        self.loss_time = 0
        self.EPS = 1e-6
