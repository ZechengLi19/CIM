import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import box_iou,nms
import json

# PCL loss
# https://arxiv.org/pdf/1807.03342.pdf
def PCL_loss(predict_cls, mat, labels):
    loss = torch.tensor(0.).cuda(device=labels.device)

    # find background index
    bg_ind = np.setdiff1d(mat[:, 0].cpu().numpy(), [0])
    if len(bg_ind) == 0:
        # without background
        bg_ind = 10000
    else:
        # with background
        assert len(bg_ind) == 1
        bg_ind = bg_ind[0]
    fg_bg_num = 1e-6
    for cluster_ind in mat.unique():
        # foreground loss
        if cluster_ind.item() != 0 and cluster_ind.item() != bg_ind:
            TFmat = (mat == cluster_ind)
            refine_tmp = predict_cls[TFmat.sum(1) != 0, :]
            col_ind = (TFmat.sum(0) != 0).float()
            refine_tmp_vector = refine_tmp.mean(0)
            fg_bg_num += refine_tmp.shape[0]
            loss += refine_tmp.shape[0] * mil_loss(refine_tmp_vector, col_ind)
        # background loss
        elif cluster_ind.item() == bg_ind:
            TFmat = (mat == cluster_ind)
            refine_tmp = predict_cls[TFmat.sum(1) != 0, :]
            gt_tmp = (mat[TFmat.sum(1) != 0, :] != 0).float()
            fg_bg_num += refine_tmp.shape[0]
            loss += refine_tmp.shape[0] * mil_loss(refine_tmp, gt_tmp)

    loss = loss / fg_bg_num
    return 12 * loss

def loss_weight_bag_loss(predict, pseudo_labels, labels, loss_weight):
    assert predict.ndim == 2
    labels = labels.squeeze()
    assert labels.ndim == 1

    # find foreground
    ind = (pseudo_labels != 0).sum(-1) != 0
    tmp_pseudo_label = (pseudo_labels != 0).float()
    assert tmp_pseudo_label.max() == 1

    # find the most discriminative proposal
    # foreground and background part
    fg_agg_value, fg_agg_index = torch.max(ind[:,None] * predict * tmp_pseudo_label,dim=0)
    # unseen classes part
    unseen_agg_value, unseen_agg_index = torch.max(predict,dim=0)

    # aggregate scores
    aggression = (fg_agg_value * labels) + (unseen_agg_value * (1 - labels))
    aggression = aggression.clamp(1e-6, 1 - 1e-6)

    # aggregate index
    label_flag = labels == 1
    aggression_index = torch.zeros_like(unseen_agg_index)
    aggression_index[label_flag] = fg_agg_index[label_flag]
    aggression_index[~label_flag] = unseen_agg_index[~label_flag]

    label_weight = loss_weight[aggression_index]
    label_weight[~label_flag] = 1

    loss = - (labels * torch.log(aggression) + (1 - labels) * torch.log(1 - aggression)) * label_weight # BCE loss

    return loss.mean()

# cal cls_loss, iou_loss
# use image label
def cls_iou_loss(cls_score, iou_score, pseudo_labels, pseudo_iou_label, loss_weights, labels, del_iou_branch=False):
    pseudo_iou_label = pseudo_iou_label.flatten()
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
    iou_score = iou_score.clamp(1e-6, 1 - 1e-6)

    label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
    label_tmp[:, 1:] = labels

    ind = (pseudo_labels != 0).sum(-1) != 0

    # for ablation study
    if del_iou_branch:
        bag_loss = loss_weight_bag_loss(cls_score, pseudo_labels, label_tmp, loss_weights)
    # CIM default setting
    else:
        # class-agnostic
        if iou_score.shape[-1] == 1:
            temp_op_score = torch.concat((cls_score[:,0:1], cls_score[:,1:] * iou_score),dim=1)
            bag_loss = loss_weight_bag_loss(temp_op_score, pseudo_labels, label_tmp, loss_weights)
        # class-specific
        else:
            bag_loss = loss_weight_bag_loss(cls_score*iou_score, pseudo_labels, label_tmp, loss_weights)

    cls_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    iou_loss = torch.tensor(0.).to(device=pseudo_labels.device)

    if ind.sum() != 0:
        pseudo_labels = (pseudo_labels[ind] != 0).float()
        assert pseudo_labels.max() == 1
        pseudo_iou_label = pseudo_iou_label[ind]
        cls_score = cls_score[ind]
        iou_score = iou_score[ind]
        loss_weights = loss_weights[ind]

        # cls_loss
        cls_loss = -pseudo_labels * torch.log(cls_score) * loss_weights.view(-1,1)
        cls_loss = cls_loss.sum() / pseudo_labels.sum()

        fg_ind = (pseudo_labels[:,1:] != 0).sum(-1) != 0
        if fg_ind.sum() != 0:
            fg_pseudo_labels = pseudo_labels[fg_ind]
            fg_pseudo_iou_label = pseudo_iou_label[fg_ind]
            fg_iou_score = iou_score[fg_ind]
            fg_loss_weights = loss_weights[fg_ind]

            # iou score --> class-specific
            if fg_iou_score.shape[-1] == fg_pseudo_labels.shape[-1]:
                fg_iou_score = (fg_pseudo_labels * fg_iou_score).sum(-1)
            # iou score --> class-agnostic
            elif fg_iou_score.shape[-1] == 1:
                fg_iou_score = fg_iou_score.squeeze()
            else:
                raise NotImplementedError("Please check shape of fg_iou_score")

            iou_loss = nn.functional.smooth_l1_loss(fg_iou_score, fg_pseudo_iou_label,reduction="none") * fg_loss_weights
            iou_loss = iou_loss.sum() / fg_pseudo_labels.sum()

        else:
            iou_loss = torch.tensor(0.).to(device=pseudo_labels.device)

    return cls_loss, iou_loss, bag_loss

def mil_loss(cls_score, labels, loss_weight=None):
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
    labels = labels.clamp(0, 1)
    loss = -labels * torch.log(cls_score) - (1 - labels) * torch.log(1 - cls_score)
    if loss_weight != None:
        loss = loss * loss_weight

    return loss.mean()

def mil_bag_loss(predict_cls, predict_det,labels):
    pred = predict_cls * predict_det
    pred = torch.sum(pred,dim=0,keepdim=True)
    pred = pred.clamp(1e-6, 1 - 1e-6)

    # background in pred
    if pred.shape[-1]-1 == labels.shape[-1]:
        label_tmp = labels.new_ones(labels.shape[0], labels.shape[1] + 1)
        label_tmp[:, 1:] = labels # padding background

    # background not in pred
    else:
        label_tmp = labels.new_ones(labels.shape[0], labels.shape[1])
        label_tmp[:, 0:] = labels

    loss = - (label_tmp * torch.log(pred) + (1 - label_tmp) * torch.log(1 - pred)) # BCE loss

    return loss.mean()

class cls_iou_model(nn.Module):
    def __init__(self, dim_in, dim_out, refine_times, class_agnostic=False):
        super(cls_iou_model, self).__init__()

        # Anti-noise branch
        self.classifier = nn.Linear(dim_in,dim_out)
        self.detector = nn.Linear(dim_in, dim_out)
        ######

        # Refinement branches
        # learn the classification score
        self.refine_cls = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(refine_times)])

        # learn the iou score
        self.refine_iou = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(refine_times)])
        #######

    def detectron_weight_mapping(self):
        detectron_weight_mapping = dict()
        for name, _ in self.named_parameters():
            detectron_weight_mapping[name] = name

        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    # input backbone feature
    def forward(self, seg_feature):
        if seg_feature.dim() == 4:
            seg_feature = seg_feature.squeeze(3).squeeze(2)

        # Anti-noise branch forward
        predict_cls = self.classifier(seg_feature)
        predict_cls = nn.functional.softmax(predict_cls,dim=-1)

        predict_det = self.detector(seg_feature)
        predict_det = nn.functional.softmax(predict_det,dim=0)
        ########
        
        refine_cls_score = []
        refine_iou_score = []

        # Refinement branches forward
        for cls_layer, iou_layer in zip(self.refine_cls, self.refine_iou):
            cls_score = cls_layer(seg_feature)
            cls_score = nn.functional.softmax(cls_score, dim=-1)
            refine_cls_score.append(cls_score)

            iou_score = iou_layer(seg_feature)
            iou_score = F.sigmoid(iou_score)
            refine_iou_score.append(iou_score)
        #########
        return predict_cls, predict_det, refine_cls_score, refine_iou_score


class CIM_layer(nn.Module):
    def __init__(self, p_seed=0.1, cls_thr=0.25, iou_thr=0.5, con_thr=0.85, Anti_noise_sampling=True):
        super(CIM_layer, self).__init__()
        self.p_seed = p_seed
        self.cls_thr = cls_thr
        self.nms_thr = cls_thr # nms_thr uses same value of cls_thr
        self.iou_thr = iou_thr
        self.con_thr = con_thr
        self.Anti_noise_sampling = Anti_noise_sampling

        print("CIM_layer--> p_seed:{}, iou_thr: {}, cls_thr/nms_thr: {}".format(p_seed, iou_thr, cls_thr))
        print("Anti_noise_sampling: {}".format(Anti_noise_sampling))

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
                if iou < self.nms_thr:
                    return True
                else:
                    return False

            instance_list = list(filter(iou_filter, instance_list))

        return selected_instances_id

    @torch.no_grad()
    def MIST_label(self, preds, rois, label, iou_map=None):
        if label.dim() != 1:
            label = label.squeeze()

        assert label.dim() == 1
        assert label.shape[-1] == 20 or label.shape[-1] == 80

        # bg remove
        preds = (preds if preds.shape[-1] == label.shape[-1] else preds[:, 1:]).clone()  # remove background class if present
        keep_count = int(np.ceil(self.p_seed * preds.shape[0]))
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
                keep_nms_idx = nms(keep_rois, keep_cls_prob, self.nms_thr)

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
    def CIM_label(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None):
        if label.dim() != 1:
            label = label.squeeze()

        assert label.dim() == 1
        assert label.shape[-1] == 20 or label.shape[-1] == 80

        # remove background
        predict_cls = (predict_cls[:, 1:] if predict_cls.shape[-1]-1 == label.shape[-1] else predict_cls).clone()  # remove background class if present
        predict_det = (predict_det[:, 1:] if predict_det.shape[-1]-1 == label.shape[-1] else predict_det).clone()  # remove background class if present

        preds = predict_cls * predict_det

        keep_count = int(np.ceil(self.p_seed * predict_cls.shape[0]))
        klasses = label.nonzero(as_tuple=True)[0]
        # generate one hot label
        gt_labels = torch.zeros((predict_cls.shape[0], label.shape[-1] + 1), dtype=predict_cls.dtype, device=predict_cls.device)
        gt_weights = -torch.ones((predict_cls.shape[0],), dtype=predict_cls.dtype, device=predict_cls.device)
        # filter out big proposals
        asy_iou_flag = torch.sum(asy_iou_map > self.con_thr, dim=-1, keepdim=True) < 0.9 * asy_iou_map.shape[-1]

        for c in klasses:
            cls_prob_tmp = predict_cls[:, c]
            # class-specific
            if predict_det.shape[-1] == label.shape[-1]:
                det_prob_tmp = predict_det[:, c]
            # class-agnostic
            elif predict_det.shape[-1] == 1:
                det_prob_tmp = predict_det[:, 0]
            else:
                raise NotImplementedError("Detector only supports class-specific and class-agnostic methods")

            preds_tmp = preds[:, c]

            # Step1: selecting seeds
            keep_sort_idx = cls_prob_tmp.argsort(descending=True)[:keep_count]
            # top p percent of proposals
            keep_rois = rois[keep_sort_idx]
            keep_cls_prob = cls_prob_tmp[keep_sort_idx]

            # NMS
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
                keep_nms_idx = nms(keep_rois, keep_cls_prob, self.nms_thr)

            # mapping index to original index
            keep_nms_idx = keep_sort_idx[keep_nms_idx]
            ###########

            # Step2: mining pseudo ground truth
            assert asy_iou_map != None
            # Note: asy_iou_map[i,j] indicates to what extent the i-th proposal contain the j-th proposal
            temp_asy_iou_map = asy_iou_map[:, keep_nms_idx]
            temp_asy_iou_map = temp_asy_iou_map > self.con_thr

            # filter out big proposals
            flag = temp_asy_iou_map * asy_iou_flag
            if flag.sum() != 0:
                flag = flag[:, torch.sum(flag, dim=0) > 0]
                res_det = flag * det_prob_tmp[:, None]
                res_idx = torch.argmax(res_det, dim=0)
                res_idx = torch.unique(res_idx)

                is_higher_scoring_class = preds_tmp[res_idx] > gt_weights[res_idx]
                if is_higher_scoring_class.sum() > 0:
                    keep_idxs = res_idx[is_higher_scoring_class]
                    gt_labels[keep_idxs, :] = 0
                    gt_labels[keep_idxs, c + 1] = 1
                    gt_weights[keep_idxs] = preds_tmp[keep_idxs]

        gt_idxs = torch.sum(gt_labels, dim=-1) > 0
        gt_boxes, gt_labels, gt_weights = rois[gt_idxs], gt_labels[gt_idxs], gt_weights[gt_idxs]

        return gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag

    @torch.no_grad()
    def forward(self, predict_cls, predict_det, rois, labels, iou_map=None, asy_iou_map=None, using_CIM = True):
        if rois.ndim == 3:
            rois = rois.squeeze(0)
        rois = rois[:,1:]

        # using CIM strategy
        if using_CIM:
            gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag = self.CIM_label(predict_cls, predict_det, rois, labels, iou_map, asy_iou_map)
        # using MIST strategy
        # https://arxiv.org/pdf/2004.04725.pdf
        ######
        else:
            if predict_det!= None:
                preds = predict_cls * predict_det
            else:
                preds = predict_cls

            gt_boxes, gt_labels, gt_weights, gt_idxs = self.MIST_label(preds, rois, labels, iou_map)

        if gt_idxs.sum() == 0:
            return None, None, None, None

        if iou_map == None:
            overlaps = box_iou(rois, gt_boxes)
        else:
            overlaps = iou_map[:, gt_idxs]

        # Anti-noise sampling
        if self.Anti_noise_sampling:
            if labels.dim() != 1:
                label = labels.squeeze()
            else:
                label = labels

            assert label.dim() == 1
            assert label.shape[-1] == 20 or label.shape[-1] == 80

            klasses = label.nonzero(as_tuple=True)[0]

            inds = torch.ones_like(gt_labels[:, 0], device=gt_labels.device)

            for c in klasses:
                # skip background
                class_idx = torch.nonzero(gt_labels[:, c + 1] == 1).flatten().cpu().numpy()
                if len(class_idx) == 0:
                    continue

                prob = gt_weights[class_idx].cpu().numpy()
                # sampling with replacement
                sampled_class_idx = np.random.choice(class_idx, size=len(class_idx), replace=True,
                                                     p=prob / prob.sum())
                sampled_class_idx = np.unique(sampled_class_idx)

                # clean original labels
                inds[class_idx] = 0
                # relabel
                inds[sampled_class_idx] = 1

            # keep some gt after sampling
            inds = inds == 1
            gt_weights = gt_weights[inds]
            gt_labels = gt_labels[inds, :]
            gt_boxes = gt_boxes[inds, :]
            overlaps = overlaps[:, inds]
        ################

        # assign pseudo labels to all proposals based on the IoU
        max_overlap_v, max_overlap_idx = torch.max(overlaps,dim=-1)

        pseudo_labels = gt_labels[max_overlap_idx]
        loss_weights = gt_weights[max_overlap_idx]
        pseudo_iou_label = max_overlap_v

        # filter out irrelevant proposals (without overlap with gt)
        ignore_inds = max_overlap_v == 0
        pseudo_labels[ignore_inds, :] = 0
        loss_weights[ignore_inds] = 0

        # assign background class
        bg_inds = (max_overlap_v < self.cls_thr) * ~ignore_inds
        pseudo_labels[bg_inds,:] = 0
        pseudo_labels[bg_inds,0] = 1

        try:
            big_proposal = ~asy_iou_flag
            pseudo_labels[big_proposal, :] = 0
            pseudo_labels[big_proposal, 0] = 1
        except:
            pass

        pseudo_iou_label[pseudo_iou_label > self.iou_thr] = 1
        pseudo_iou_label[pseudo_iou_label <= self.iou_thr] = 0

        return pseudo_labels, pseudo_iou_label, loss_weights