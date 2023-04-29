import numpy as np
import numpy.random as npr
import cv2

from core.config import cfg
import utils.blob as blob_utils
from PIL import Image


def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data', 'rois', 'masks', 'labels','gtrois', 'mat', "True_rois","iou_label", 'peak_score']
    return blob_names


    
def get_minibatch(roidb, num_classes, flag):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob, im_scales = _get_image_blob(roidb, flag)

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    blobs['data'] = im_blob
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    True_rois_blob = np.zeros((0, 5), dtype=np.float32)

    masks_blob = np.zeros((0, cfg.FAST_RCNN.MASK_SIZE, cfg.FAST_RCNN.MASK_SIZE), dtype=np.float32)
    
    mat_blob = np.zeros((0, cfg.MODEL.NUM_CLASSES + 1), dtype=np.float32)
    iou_label_blob = np.zeros((0, 1), dtype=np.float32)
    peak_score_blob = np.zeros((0, 1), dtype=np.float32)
    gtbox_blob = np.zeros((0, 6), dtype=np.float32)
    labels_blob = np.zeros((0, num_classes), dtype=np.float32)

    num_images = len(roidb)
    for im_i in range(num_images):
        labels, im_rois, gt_rois, True_rois = _sample_rois(roidb[im_i], num_classes)
        mat_blob_this_image = roidb[im_i]['mat']
        iou_label_blob_this_image = roidb[im_i]['iou_label']
        peak_score_blob_this_image = roidb[im_i]['peak_score']
        img_path = roidb[im_i]['image']

        
        #if not mat_blob_this_image:
        #    mat_blob_this_image = np.zeros((len(im_rois), cfg.MODEL.NUM_CLASSES + 1 ), dtype=np.float32)
        
        
        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i])
        True_rois = _project_im_rois(True_rois, im_scales[im_i])

        gt_rois = _project_im_gtrois(gt_rois, im_scales[im_i])

        batch_ind = im_i * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))
        True_rois_blob_this_image = np.hstack((batch_ind, True_rois))
        
        masks_blob_this_image =  roidb[im_i]['masks'].astype(np.float32)
        
        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(rois_blob_this_image * cfg.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True,
                                            return_inverse=True)
            if cfg.iou_label:
                iou_label_blob_this_image = iou_label_blob_this_image[index, :]

            if cfg.SCORE:
                peak_score_blob_this_image = peak_score_blob_this_image[index, :]

            rois_blob_this_image = rois_blob_this_image[index, :]
            True_rois_blob_this_image = True_rois_blob_this_image[index, :]

            mat_blob_this_image = mat_blob_this_image[index, :]
            masks_blob_this_image = masks_blob_this_image[index, :, :]
        else:
            index = np.arange(rois_blob_this_image.shape[0])

        rois_blob = np.vstack((rois_blob, rois_blob_this_image))
        True_rois_blob = np.vstack((True_rois_blob, True_rois_blob_this_image))
        masks_blob = np.vstack((masks_blob, masks_blob_this_image))
        mat_blob = np.vstack((mat_blob, mat_blob_this_image))
        iou_label_blob = np.vstack((iou_label_blob,iou_label_blob_this_image))
        peak_score_blob = np.vstack((peak_score_blob, peak_score_blob_this_image))
        batch_ind = im_i * np.ones((gt_rois.shape[0], 1))
        gt_rois_blob_this_image = np.hstack((batch_ind, gt_rois))
        gtbox_blob = np.vstack((gtbox_blob, gt_rois_blob_this_image))

        # Add to labels blob
        labels_blob = np.vstack((labels_blob, labels))
    try:
        blobs['index'] = index
        blobs['rois'] = rois_blob
        blobs['True_rois'] = True_rois_blob
        blobs['masks'] = masks_blob
        blobs['labels'] = labels_blob
        blobs['gtrois'] = gtbox_blob
        blobs['mat'] = mat_blob
        blobs['iou_label'] = iou_label_blob
        blobs['peak_score'] = peak_score_blob
        blobs['path'] = img_path
    except:
        blobs['rois'] = rois_blob
        blobs['True_rois'] = True_rois_blob
        blobs['masks'] = masks_blob
        blobs['labels'] = labels_blob
        blobs['gtrois'] = gtbox_blob
        blobs['mat'] = mat_blob
        blobs['iou_label'] = iou_label_blob
        blobs['peak_score'] = peak_score_blob
        blobs['path'] = img_path

   
    return blobs, True


def _sample_rois(roidb, num_classes):
    """Generate a random sample of RoIs"""
    labels = roidb['gt_classes']
    rois = roidb['boxes']
    gt_rois = roidb['gt_boxes']
    True_rois = roidb['True_rois']

    if cfg.TRAIN.BATCH_SIZE_PER_IM > 0:  # 4096
        batch_size = cfg.TRAIN.BATCH_SIZE_PER_IM
    else:
        batch_size = np.inf
    if batch_size < rois.shape[0]:
        rois_inds = npr.permutation(rois.shape[0])[:batch_size]
        rois = rois[rois_inds, :]
        True_rois = True_rois[rois_inds, :]

    return labels.reshape(1, -1), rois, gt_rois, True_rois


def _get_image_blob(roidb, flag):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        if flag == "org":
            im = cv2.imread(roidb[i]['image'])
            assert im is not None, \
                'Failed to read image \'{}\''.format(roidb[i]['image'])
            # If NOT using opencv to read in images, uncomment following lines
            # if len(im.shape) == 2:
            #     im = im[:, :, np.newaxis]
            #     im = np.concatenate((im, im, im), axis=2)
            # # flip the channel, since the original one using cv2
            # # rgb -> bgr
            # im = im[:, :, ::-1]
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            im, im_scale = blob_utils.prep_im_for_blob(
                im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE,flag)
            im_scales.append(im_scale[0])
            processed_ims.append(im[0])
        elif flag == "ToTensor":
            im = cv2.imread(roidb[i]['image'])
            assert im is not None, \
                'Failed to read image \'{}\''.format(roidb[i]['image'])
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            im, im_scale = blob_utils.prep_im_for_blob(
                im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE ,flag)
            im_scales.append(im_scale[0])
            processed_ims.append(im[0])
        else:
            print("minibatch error!!")
            raise AssertionError

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    im_rois_tmp = im_rois.copy()
    rois = im_rois_tmp * im_scale_factor
    return rois

def _project_im_gtrois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois =  im_rois.copy()
    rois[:,:4] = rois[:,:4]* im_scale_factor
    
    return rois