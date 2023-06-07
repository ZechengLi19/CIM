import pickle
from functools import wraps
import importlib
import logging
import torch
import torch.nn as nn
from core.config import cfg
from ops import RoIPool, RoIAlign
import modeling.heads as heads
import utils.vgg_weights_helper as vgg_utils
import utils.hrnet_weights_helper as hrnet_utils
import os

logger = logging.getLogger(__name__)

def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise

def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper

def testing_function(predict_cls, predict_det, ref_cls_score, ref_iou_score,return_dict):
    res = []
    for cls_score, iou_score in zip(ref_cls_score, ref_iou_score):
        preds = cls_score * iou_score
        res.append(preds[:, 1:])

    return_dict['refine_score'] = res

    return return_dict


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        cls_num = cfg.MODEL.NUM_CLASSES + 1

        # feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(self.Conv_Body.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)

        step_rate = cfg.step_rate

        self.cls_iou_model = heads.cls_iou_model(self.Box_Head.dim_out, cls_num, cfg.REFINE_TIMES,
                                                     class_agnostic=False)
        self.mist_layer_list = []
        for ref_time in range(cfg.REFINE_TIMES):
            self.mist_layer_list.append(heads.mist_layer(portion=cfg.topk,
                                                         full_thr=0.5 + step_rate * ref_time,
                                                         iou_thr=0.25 + step_rate * ref_time,
                                                         sample=cfg.easy_case_mining,
                                                         ))

        self.diffuse_mode = [True, True, True]

        print("diffuse_mode: ")
        print(self.diffuse_mode)

        self._init_modules()

    def _init_modules(self):
        if cfg.VGG_CLS_FEATURE:
            if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
                vgg_utils.load_pretrained_imagenet_weights(self)
        # resnet using pre-trained weight from torch
        # if not cfg.ResNet_CLS_FEATURE:
        #     if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
        #         resnet_utils.load_pretrained_imagenet_weights(self)
        if cfg.HRNET_CLS_FEATURE:
            if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
                hrnet_utils.load_pretrained_imagenet_weights(self)
        if cfg.TRAIN.FREEZE_CONV_BODY:  # false
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, rois, masks, labels, gtrois, mat, path=None, index=None):
        with torch.set_grad_enabled(self.training):
            # print(index)
            im_data = data
            if self.training:
                index = index.squeeze(dim=0).type(im_data.dtype)
                rois = rois.squeeze(dim=0).type(im_data.dtype)
                masks = masks.squeeze(dim=0).type(im_data.dtype)
                labels = labels.squeeze(dim=0).type(im_data.dtype)
                mat = mat.squeeze(dim=0).type(im_data.dtype)

            return_dict = {}  # A dict to collect return variables

            blob_conv = self.Conv_Body(im_data)

            # if not self.training:
            return_dict['blob_conv'] = blob_conv

            masks.requires_grad = False

            seg_x = self.Box_Head(blob_conv, rois, masks.detach())

            file_name = os.path.splitext(os.path.split(path)[1])[0]

            iou_dir = cfg.iou_dir
            asy_iou_dir = cfg.asy_iou_dir

            predict_cls, predict_det, ref_cls_score, ref_iou_score = self.cls_iou_model(seg_x)
            iou_map = None
            asy_iou_map = None

            if self.training:
                index = index.long()
                try:
                    iou_map = pickle.load(open(os.path.join(iou_dir, file_name + ".pkl"), "rb"))
                    iou_map = torch.tensor(iou_map, device=labels.device)[index][:, index]
                except:
                    print("iou_map lose " + os.path.join(iou_dir, file_name + ".pkl"))
                    raise AssertionError

                try:
                    asy_iou_map = pickle.load(open(os.path.join(asy_iou_dir, file_name + ".pkl"), "rb"))
                    asy_iou_map = torch.tensor(asy_iou_map, device=labels.device)[index][:, index]
                except:
                    print("asy_iou_map lose " + os.path.join(asy_iou_dir, file_name + ".pkl"))
                    raise AssertionError

                return_dict['losses'] = {}
                return_dict['losses']['bag_loss'] = torch.tensor(0, dtype=torch.float32, device=seg_x.device)
                return_dict['losses']['cls_stage1_loss'] = torch.tensor(0, dtype=torch.float32, device=seg_x.device)

                return_dict['losses']['ind_cls_loss'] = torch.tensor(0, dtype=torch.float32, device=seg_x.device)
                return_dict['losses']['ind_iou_loss'] = torch.tensor(0, dtype=torch.float32, device=seg_x.device)

                for i, (cls_score, iou_score, mist_layer) in enumerate(zip(ref_cls_score, ref_iou_score, self.mist_layer_list)):
                    # follow WSDDN
                    lmda = 3 if i == 0 else 1
                    #########

                    if i == 0:
                        pseudo_labels, pseudo_iou_label, loss_weights, group_assign = mist_layer(predict_cls,
                                                                                           predict_det,
                                                                                           rois, labels, iou_map,
                                                                                           asy_iou_map,
                                                                                           diffuse=self.diffuse_mode[i])

                    else:
                        pseudo_labels, pseudo_iou_label, loss_weights, group_assign = mist_layer(ref_cls_score[i - 1],
                                                                                           ref_iou_score[i - 1],
                                                                                           rois, labels, iou_map,
                                                                                           asy_iou_map,
                                                                                           diffuse=self.diffuse_mode[i])

                    if pseudo_labels == None:
                        continue

                    pseudo_labels = pseudo_labels.detach()
                    pseudo_iou_label = pseudo_iou_label.detach()
                    loss_weights = lmda * loss_weights.detach()

                    ind_cls_loss, ind_iou_loss, f_ind_cls_loss, f_ind_iou_loss, bag_loss = heads.cal_cls_iou_loss_function_full(cls_score, iou_score, pseudo_labels, pseudo_iou_label,loss_weights, labels)

                    return_dict['losses']['ind_cls_loss'] += ind_cls_loss.clone()
                    return_dict['losses']['ind_iou_loss'] += 3 * ind_iou_loss.clone()
                    return_dict['losses']['bag_loss'] += bag_loss.clone()

                return_dict['losses']['bag_loss'] += heads.mil_bag_loss(predict_cls, predict_det, labels)
                cls_loss, _ = heads.graph_two_Loss_mean(predict_cls, mat, labels)
                return_dict['losses']['cls_stage1_loss'] += cls_loss

                for k, v in return_dict['losses'].items():
                    return_dict['losses'][k] = v.unsqueeze(0)

            else:
                # Testing
                return_dict = testing_function(predict_cls, predict_det, ref_cls_score, ref_iou_score, return_dict)

            return return_dict

    def roi_feature_transform(self, blobs_in, rois, method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if method == 'RoIPoolF':
            xform_out = RoIPool(resolution, spatial_scale)(blobs_in, rois)
        elif method == 'RoIAlign':
            xform_out = RoIAlign(
                resolution, spatial_scale, sampling_ratio)(blobs_in.contiguous(), rois.contiguous())

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    try:
                        child_map, child_orphan = m_child.detectron_weight_mapping()
                        d_orphan.extend(child_orphan)
                        for key, value in child_map.items():
                            new_key = name + '.' + key
                            d_wmap[new_key] = value
                    except:
                        print("model:{}, dont have detectron_weight_mapping function".format(m_child))
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value