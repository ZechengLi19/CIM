import os
import numpy as np
from six.moves import cPickle as pickle
from scipy.io import loadmat
import scipy
from PIL import Image
from pycocotools.coco import COCO
import multiprocessing
from tqdm import tqdm
from pre_tools import *

trash="./data/trash"

def generate_pkl_voc2012(imgIds, worker_id):
    train_proposals_dir = "./data/VOC2012/COB_SBD_trainaug"
    val_proposals_dir = "./data/VOC2012/COB_SBD_val"
    SBD_mask_val_proposals = dict(indexes=[], masks=[], boxes=[], scores=[])
    for index in tqdm(range(len(imgIds))):
        img_id = imgIds[index]

        s = str(int(img_id))
        file_name = s[:4] + '_' + s[4:]

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
    pickle.dump(SBD_mask_val_proposals, open(os.path.join(trash,"voc_{}.pkl".format(worker_id)), 'wb'), pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    mask_size = 7
    max_proposal_num = 200
    worker = 48

    # data
    for i in range(2):
        if i == 0:
            label_file = "./data/VOC2012/annotations/voc_2012_trainaug.json"
            cob_propsal_file = "./data/cob/voc_2012_trainaug.pkl"

        else:
            label_file = "./data/VOC2012/annotations/voc_2012_val.json"
            cob_propsal_file = "./data/cob/voc_2012_val.pkl"

        cocoGt = COCO(label_file)
        imgIds = sorted(cocoGt.getImgIds())

        per_len = int(len(imgIds)/worker)

        jobs = []
        for worker_id in range(worker):
            if worker_id + 1 != worker:
                p = multiprocessing.Process(target=generate_pkl_voc2012, args=(imgIds[worker_id * per_len:(worker_id + 1) * per_len], worker_id))

            else:
                p = multiprocessing.Process(target=generate_pkl_voc2012, args=(imgIds[worker_id * per_len:], worker_id))

            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()

        res = dict(indexes=[], masks=[], boxes=[], scores=[])
        worker_id = 0
        while(worker_id != worker):
            path = os.path.join(trash, "voc_{}.pkl".format(worker_id))
            try:
                result = pickle.load(open(path, 'rb'))

                for key in res.keys():
                    res[key] += result[key]

                os.remove(path)

                worker_id += 1
            except:
                pass

        print("imgs len: "+str(len(imgIds)))

        for key in res.keys():
            print("{} len: ".format(key) + str(len(res[key])))

        pickle.dump(res, open(cob_propsal_file,'wb'), pickle.HIGHEST_PROTOCOL)
