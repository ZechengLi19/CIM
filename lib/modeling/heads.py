import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import box_iou,nms
import json

def id_2_clsname(annotation_file_path):
    with open(annotation_file_path,"r") as f:
        content = f.readlines()
    json_dict = json.loads(content[0])
    cls_name_map = [cat["name"] for cat in json_dict["categories"]]
    cls_id_map = [cat["id"] for cat in json_dict["categories"]]

    return cls_name_map,cls_id_map

# PCL loss
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
    return 12 * loss1, loss2

def loss_weight_bag_loss(predict,pseudo_labels,label_tmp,loss_weight):
    assert predict.ndim == 2
    label_tmp = label_tmp.squeeze()
    assert label_tmp.ndim == 1

    ind = (pseudo_labels != 0).sum(-1) != 0
    tmp_pseudo_label = (pseudo_labels != 0).float()
    assert tmp_pseudo_label.max() == 1

    ind_agg_value, ind_agg_index = torch.max(ind[:,None] * predict * tmp_pseudo_label,dim=0)
    agg_value, agg_index = torch.max(predict,dim=0)

    aggression = (ind_agg_value * label_tmp) + (agg_value * (1 - label_tmp))
    aggression = aggression.clamp(1e-6, 1 - 1e-6)

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

    if del_iou_branch:
        bag_loss = loss_weight_bag_loss(cls_score, pseudo_labels, label_tmp, loss_weights)

    else:
        if iou_score.shape[-1] == 1:
            temp_op_score = torch.concat((cls_score[:,0:1], cls_score[:,1:] * iou_score),dim=1)
            bag_loss = loss_weight_bag_loss(temp_op_score, pseudo_labels, label_tmp, loss_weights)
        else:
            bag_loss = loss_weight_bag_loss(cls_score*iou_score, pseudo_labels, label_tmp, loss_weights)

    ind_cls_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    ind_iou_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    f_ind_cls_loss = torch.tensor(0.).to(device=pseudo_labels.device)
    f_ind_iou_loss = torch.tensor(0.).to(device=pseudo_labels.device)

    if ind.sum() != 0:
        pseudo_labels = (pseudo_labels[ind] != 0).float()
        assert pseudo_labels.max() == 1
        pseudo_iou_label = pseudo_iou_label[ind]
        cls_score = cls_score[ind]
        iou_score = iou_score[ind]
        loss_weights = loss_weights[ind]

        # cls_loss
        ind_cls_loss = -pseudo_labels * torch.log(cls_score) * loss_weights.view(-1,1)
        ind_cls_loss = ind_cls_loss.sum() / pseudo_labels.sum()

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
                raise AssertionError

            ind_iou_loss = nn.functional.smooth_l1_loss(fg_iou_score, fg_pseudo_iou_label,reduction="none") * fg_loss_weights
            ind_iou_loss = ind_iou_loss.sum() / fg_pseudo_labels.sum()

        else:
            ind_iou_loss = torch.tensor(0.).to(device=pseudo_labels.device)

    return ind_cls_loss, ind_iou_loss, f_ind_cls_loss, f_ind_iou_loss, bag_loss

def mil_losses(cls_score, labels,loss_weight=None):
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)  # min-max  [1e-6,1 - 1e-6]
    labels = labels.clamp(0, 1)
    loss = -labels * torch.log(cls_score) - (1 - labels) * torch.log(1 - cls_score)
    if loss_weight != None:
        loss = loss * loss_weight

    return loss.mean()

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

class cls_iou_model(nn.Module):
    def __init__(self, dim_in, dim_out,ref_num,class_agnostic=False):
        super(cls_iou_model, self).__init__()

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

    def forward(self, seg_feature):
        if seg_feature.dim() == 4:
            seg_feature = seg_feature.squeeze(3).squeeze(2)

        # WSDDN
        predict_cls = self.classifier(seg_feature)
        predict_cls = nn.functional.softmax(predict_cls,dim=-1)

        inner_predict_det = self.inner_detection(seg_feature)

        predict_det = inner_predict_det

        predict_det = nn.functional.softmax(predict_det,dim=0)

        ref_cls_score = []
        ref_iou_score = []

        for i, (cls_layer, iou_layer) in enumerate(zip(self.ref_cls,self.ref_iou)):
            cls_score = cls_layer(seg_feature)
            cls_score = nn.functional.softmax(cls_score, dim=-1)
            ref_cls_score.append(cls_score)

            iou_score = iou_layer(seg_feature)
            iou_score = F.sigmoid(iou_score)
            ref_iou_score.append(iou_score)

        return predict_cls, predict_det, ref_cls_score, ref_iou_score


class CIM_layer(nn.Module):
    def __init__(self, portion=0.1, full_thr=0.5, iou_thr=0.25, asy_iou_th=0.85,sample=False):
        super(CIM_layer, self).__init__()
        self.portion = portion
        self.full_thr = full_thr
        self.iou_th = iou_thr
        self.asy_iou_th = asy_iou_th
        self.sample = sample

        print("CIM_layer--> portion:{}, full_thr: {}, iou_thr: {}".format(portion,full_thr,iou_thr))
        print("sample:{}".format(sample))

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
    def CIM_label(self, predict_cls, predict_det, rois, label, iou_map=None, asy_iou_map=None):
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
        # filter out big proposals
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
    def forward(self, predict_cls, predict_det, rois, labels, iou_map=None, asy_iou_map=None, diffuse = False):
        if rois.ndim == 3:
            rois = rois.squeeze(0)
        rois = rois[:,1:]

        if diffuse:
            gt_boxes, gt_labels, gt_weights, gt_idxs, asy_iou_flag = self.CIM_label(predict_cls, predict_det, rois, labels, iou_map, asy_iou_map)
        else:
            if predict_det!= None:
                preds = predict_cls * predict_det
            else:
                preds = predict_cls

            gt_boxes, gt_labels, gt_weights, gt_idxs = self.mist_label(preds,rois,labels,iou_map,asy_iou_map)

        if gt_idxs.sum() == 0:
            return None, None, None, None

        if iou_map == None:
            overlaps = box_iou(rois, gt_boxes)
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

        ignore_inds = max_overlap_v == 0
        pseudo_labels[ignore_inds, :] = 0
        loss_weights[ignore_inds] = 0

        bg_inds = (max_overlap_v < self.iou_th) * ~ignore_inds
        pseudo_labels[bg_inds,:] = 0
        pseudo_labels[bg_inds,0] = 1

        try:
            big_proposal = ~asy_iou_flag
            pseudo_labels[big_proposal, :] = 0
            pseudo_labels[big_proposal, 0] = 1
        except:
            pass

        pseudo_iou_label[pseudo_iou_label > self.full_thr] = 1
        pseudo_iou_label[pseudo_iou_label <= self.full_thr] = 0

        group_assign = max_overlap_idx + 1
        group_assign[bg_inds] = -1
        group_assign[ignore_inds] = -2
        group_assign = group_assign[:,None] * pseudo_labels

        return pseudo_labels, pseudo_iou_label, loss_weights, group_assign



