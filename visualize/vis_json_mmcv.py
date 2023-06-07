# -*- coding: utf-8 -*-
from pycocotools import mask as COCOMask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from six.moves import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pycocotools import mask as maskUtils
import os
import argparse
from tqdm import tqdm
from mmcv_box.visualization.image import imshow_det_bboxes
import multiprocessing
import os.path as osp
import sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)
from datasets.json_inference import coco_inst_seg_eval


def colormap(rgb=False):
    color_list = np.array(
        [
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000,
            0.000, 0.447, 0.741,
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3))
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list

def id_2_clsname(annotation_file_path):
    with open(annotation_file_path,"r") as f:
        content = f.readlines()
    json_dict = json.loads(content[0])
    cls_name_map = [cat["name"] for cat in json_dict["categories"]]

    name_2_index = dict()
    for cat in json_dict["categories"]:
        name_2_index[cat["name"]] = cat["id"]

    return cls_name_map,name_2_index


def gt_dataset_mask(cocoGT, save_dir,name_mapping):
    imgIds = sorted(cocoGT.getImgIds())
    Cls_Ids = cocoGT.getCatIds()

    for index in tqdm(range(len(imgIds)), total=len(imgIds)):
        img_id = [imgIds[index]]

        path = cocoGT.loadImgs(img_id)[0]['file_name']
        if os.path.exists(os.path.join(save_dir, path)):
            continue

        gt_ann_ids = cocoGT.getAnnIds(imgIds=img_id)
        anns = cocoGT.loadAnns(gt_ann_ids)
        anns = sorted(anns, key=lambda x: x['area'],reverse=True)

        img = Image.open(os.path.join(root, path)).convert("RGB")
        img = np.array(img)

        polygons = []
        label_list = []
        box_list = []
        score_list = []
        color = []
        color_array = colormap(True)
        w_ratio = .4
        color_array = color_array * (1 - w_ratio) + w_ratio
        for idx, ann in enumerate(anns):  # ann=anns[0]
            if idx >= color_array.shape[0]:
                c = (np.random.random((1, 3)) * 0.6 + 0.2)
            else:
                c = color_array[idx][None,:]

            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:
                    rle = maskUtils.frPyObjects(ann['segmentation'], img.shape[0], img.shape[1])
                    mask = maskUtils.decode(rle).transpose(2, 0, 1)
                    polygons.append(mask)
                    for _ in range(mask.shape[0]):
                        label_list.append(Cls_Ids.index(ann['category_id']))
                        box_list.append(np.array(ann['bbox']))
                        color.append(c)

                else:
                    if type(ann['segmentation']['counts']) == list:
                        rle = maskUtils.frPyObjects([ann['segmentation']], img.shape[0], img.shape[1])
                    else:
                        rle = [ann['segmentation']]
                    mask = maskUtils.decode(rle).transpose(2,0,1)
                    if ann['iscrowd'] == 1:
                        continue
                    elif ann['iscrowd'] == 0:
                        polygons.append(mask)
                        label_list.append(Cls_Ids.index(ann['category_id']))
                        box_list.append(np.array(ann['bbox']))
                        color.append(c)

                if 'score' in ann:
                    # box_list[-1] = np.concatenate((ann['score'],box_list[-1]),axis=1)
                    score_list.append(ann['score'])
            else:
                pass

        try:
            polygons = np.concatenate(polygons, axis=0)
            box_list = np.concatenate(box_list, axis=0).reshape(-1, 4)
            box_list[:, 2:] = box_list[:, :2] + box_list[:, 2:]
            color = np.concatenate(color, axis=0)

            label_list = np.array(label_list)
            score_list = np.array(score_list).reshape(-1, 1)
            if len(score_list) != 0:
                box_list = np.concatenate((box_list, score_list), axis=1)

            gt_img = imshow_det_bboxes(img, box_list, labels=label_list, segms=polygons,
                                       class_names=name_mapping, show=False,
                                       bbox_color=color, mask_color=255 * color,
                                       font_size=18, thickness=1,
                                       alpha=0.8)

            plt.imshow(np.uint8(gt_img))
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, path), dpi=300, bbox_inches='tight', pad_inches=0)

        except:
            plt.imshow(np.uint8(img))
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, path), dpi=300, bbox_inches='tight', pad_inches=0)

    print(f'\rImage Index: {(index + 1):.0f}/{len(imgIds):.0f}  ', end='')

def dataset_mask(cocoGT, cocoDt, save_dir, imgIds=None,num=None,anno_thr=-1, name_mapping=None):
    if imgIds == None:
        imgIds = sorted(cocoGT.getImgIds())

    Cls_Ids = cocoGT.getCatIds()

    for index in tqdm(range(len(imgIds)), total=len(imgIds)):
        img_id = imgIds[index]

        gt_ann_ids = cocoGT.getAnnIds(imgIds=[img_id])
        gt_anns = cocoGT.loadAnns(gt_ann_ids)

        ann_ids = cocoDt.getAnnIds(imgIds=[img_id])
        anns = cocoDt.loadAnns(ann_ids)
        anns = sorted(anns, key=lambda x: x['score'])
        score = np.asarray([x["score"] for x in anns])

        if anno_thr == -1:
            gt_nums = len(gt_anns)
        else:
            gt_nums = len((score >= anno_thr).nonzero()[0])

        if gt_nums == 0:
            anns = []
        else:
            anns = anns[-gt_nums:]
            anns = sorted(anns, key=lambda x: x['area'],reverse=True)

        path = cocoGT.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(root, path)).convert("RGB")
        img = np.array(img)

        if len(anns) == 0:
            plt.imshow(np.uint8(img))
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, path), dpi=300, bbox_inches='tight', pad_inches=0)

        else:
            anns = sorted(anns, key=lambda x: x['area'], reverse=True)

            polygons = []
            label_list = []
            box_list = []
            score_list = []
            color = []
            color_array = colormap(True)
            w_ratio = .4
            color_array = color_array * (1 - w_ratio) + w_ratio
            for idx, ann in enumerate(anns):  # ann=anns[0]
                if idx >= color_array.shape[0]:
                    c = (np.random.random((1, 3)) * 0.6 + 0.2)
                else:
                    c = color_array[idx][None, :]

                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        rle = maskUtils.frPyObjects(ann['segmentation'], img.shape[0], img.shape[1])
                        mask = maskUtils.decode(rle).transpose(2, 0, 1)
                        polygons.append(mask)
                        for _ in range(mask.shape[0]):
                            label_list.append(Cls_Ids.index(ann['category_id']))
                            box_list.append(np.array(ann['bbox']))
                            color.append(c)

                    else:
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], img.shape[0], img.shape[1])
                        else:
                            rle = [ann['segmentation']]
                        mask = maskUtils.decode(rle).transpose(2, 0, 1)
                        if ann['iscrowd'] == 1:
                            continue
                        elif ann['iscrowd'] == 0:
                            polygons.append(mask)
                            label_list.append(Cls_Ids.index(ann['category_id']))
                            box_list.append(np.array(ann['bbox']))
                            color.append(c)
                    if 'score' in ann:
                        score_list.append(ann['score'])
                else:
                    pass

            polygons = np.concatenate(polygons, axis=0)
            label_list = np.array(label_list)
            box_list = np.concatenate(box_list, axis=0).reshape(-1,4)
            box_list[:,2:] = box_list[:,:2] + box_list[:,2:]
            score_list = np.array(score_list).reshape(-1,1)
            color = np.concatenate(color,axis=0)
            if len(score_list) != 0:
                box_list = np.concatenate((box_list,score_list),axis=1)
            gt_img = imshow_det_bboxes(img,box_list,labels=label_list,segms=polygons,
                                       class_names=name_mapping,show=False,
                                       bbox_color=color,mask_color=255 * color,
                                       font_size=18,thickness=1,
                                       alpha=0.8)

            plt.imshow(np.uint8(gt_img))
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, path), dpi=300, bbox_inches='tight',pad_inches=0)

        if num is not None:
            if index == num - 1:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="eval model")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--result_file_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--thr", type=float, default=-1)
    parser = parser.parse_args()

    if "gt" not in parser.dataset:
        if parser.dataset == "voc_val":
            label_file = "./data/VOC2012/annotations/voc_2012_val.json"
            root = './data/VOC2012/JPEGImages'

        elif parser.dataset == "voc_train":
            label_file = "./data/VOC2012/annotations/voc_2012_trainaug.json"
            root = './data/VOC2012/JPEGImages'

        elif parser.dataset == "coco_val":
            label_file = "./data/coco2017/annotations/instances_val2017.json"
            root = './data/coco2017/val2017'

        elif parser.dataset == "coco_test":
            label_file = "./data/coco2017/annotations/image_info_test-dev2017.json"
            root = './data/coco2017/test2017'

        elif parser.dataset == "coco_train":
            label_file = "./data/coco2017/annotations/instances_train2017.json"
            root = './data/coco2017/train2017'

        result_file = parser.result_file_path
        save_dir = parser.save_dir
        thr = parser.thr
        os.makedirs(save_dir, exist_ok=True)

        cocoGt = COCO(label_file)
        imgIds = sorted(cocoGt.getImgIds())

        res = json.load(open(result_file))

        try:
            if "annotations" in res.keys():
                temp_filename = os.path.join("./data/trash",
                                             './temp.json')
                with open(temp_filename, 'w') as file_obj:
                    json.dump(res['annotations'], file_obj)

                result_file = temp_filename
        except:
            pass
        cocoDt1 = cocoGt.loadRes(result_file)

        cls_name_map,name_2_index = id_2_clsname(label_file)

        worker = 12

        per_len = int(len(imgIds) / worker)

        jobs = []
        for worker_id in range(worker):
            if worker_id + 1 != worker:
                p = multiprocessing.Process(target=dataset_mask,
                                            args=(cocoGt, cocoDt1, save_dir, imgIds[worker_id * per_len:(worker_id + 1) * per_len], parser.num, thr, cls_name_map))
            else:
                p = multiprocessing.Process(target=dataset_mask,
                                            args=(cocoGt, cocoDt1, save_dir, imgIds[worker_id * per_len:], parser.num, thr, cls_name_map))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()

        print(len(os.listdir(save_dir)))

        mAP, cls_ap, cls_names = coco_inst_seg_eval(label_file, result_file)

        stack_item = []
        for key, value in cls_ap.items():
            stack_item.append(value)

        stack_item = np.concatenate(stack_item, axis=0).reshape((len(cls_ap.keys()), -1)).transpose()

        print('Class Performance(COCOAPI): ')

        for idx, _ in enumerate(cls_names):
            print("%-15s -->  %.1f, %.1f, %.1f, %.1f" % (cls_names[idx], 100 * stack_item[idx][0],
                                                         100 * stack_item[idx][1], 100 * stack_item[idx][2],
                                                         100 * stack_item[idx][3]))

        print('Performance(COCOAPI): ')
        for k, v in mAP.items():
            print('mAP@%s: %.1f' % (k, 100 * v))

    elif parser.dataset == "voc_val_gt":
        label_file = "./data/VOC2012/annotations/voc_2012_val.json"
        root = './data/VOC2012/JPEGImages'

        save_dir = parser.save_dir
        os.makedirs(save_dir,exist_ok=True)

        cls_name_map,name_2_index = id_2_clsname(label_file)


        cocoGt = COCO(label_file)
        gt_dataset_mask(cocoGt, save_dir,cls_name_map)

    elif parser.dataset == "voc_train_gt":
        label_file = "./data/VOC2012/annotations/voc_2012_trainaug.json"
        root = './data/VOC2012/JPEGImages'

        save_dir = parser.save_dir
        os.makedirs(save_dir,exist_ok=True)

        cls_name_map,name_2_index = id_2_clsname(label_file)

        cocoGt = COCO(label_file)
        gt_dataset_mask(cocoGt, save_dir,cls_name_map)

    elif parser.dataset == "coco_val_gt":
        label_file = "./data/coco2017/annotations/instances_val2017.json"
        root = './data/coco2017/val2017'

        save_dir = parser.save_dir
        os.makedirs(save_dir,exist_ok=True)

        cls_name_map,name_2_index = id_2_clsname(label_file)

        cocoGt = COCO(label_file)
        gt_dataset_mask(cocoGt, save_dir,cls_name_map)

    elif parser.dataset == "coco_train_gt":
        label_file = "./data/coco2017/annotations/instances_train2017.json"
        root = './data/coco2017/train2017'
        save_dir = parser.save_dir
        os.makedirs(save_dir,exist_ok=True)

        cls_name_map,name_2_index = id_2_clsname(label_file)

        cocoGt = COCO(label_file)
        gt_dataset_mask(cocoGt, save_dir,cls_name_map)

    else:
        raise NotImplementedError
