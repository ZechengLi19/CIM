# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""blob helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import cPickle as pickle
import numpy as np
import cv2

from core.config import cfg
from torchvision import transforms


def get_image_blob(im, target_scale, target_max_size,flag):
    """Convert an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale (float): image scale (target size) / (original size)
        im_info (ndarray)
    """
    processed_im, im_scale = prep_im_for_blob(
        im, cfg.PIXEL_MEANS, [target_scale], target_max_size, flag
    )
    blob = im_list_to_blob(processed_im)

    return blob, im_scale


def im_list_to_blob(ims):
    """Convert a list of images into a network input. Assumes images were
    prepared using prep_im_for_blob or equivalent: i.e.
      - BGR channel order
      - pixel means subtracted
      - resized to the desired input size
      - float32 numpy ndarray format
    Output is a 4D HCHW tensor of the images concatenated along axis 0 with
    shape.
    """
    if not isinstance(ims, list):
        ims = [ims]
    max_shape = get_max_shape([im.shape[:2] for im in ims])

    num_images = len(ims)
    blob = np.zeros(
        (num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def get_max_shape(im_shapes):
    """Calculate max spatial size (h, w) for batching given a list of image shapes
    """
    max_shape = np.array(im_shapes).max(axis=0)
    assert max_shape.size == 2
    # Pad the image so they can be divisible by a stride
    return max_shape


def prep_im_for_blob(im, pixel_means, target_sizes, max_size, flag):
    """Prepare an image for use as a network input blob. Specially:
      - Subtract per-channel pixel mean
      - Convert to float32
      - Rescale to each of the specified target size (capped at max_size)
    Returns a list of transformed images, one for each target size. Also returns
    the scale factors that were used to compute each returned image.
    """
    if flag == "org":
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        ims = []
        im_scales = []
        for target_size in target_sizes:
            im_scale = get_target_scale(im_size_min, im_size_max, target_size, max_size)
            im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                                    interpolation=cv2.INTER_LINEAR)
            ims.append(im_resized)
            im_scales.append(im_scale)
    elif flag == "ToTensor":
        im = im.astype(np.float32, copy=False)
        # im -= pixel_means
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        ims = []
        im_scales = []
        for target_size in target_sizes:
            im_scale = get_target_scale(im_size_min, im_size_max, target_size, max_size)
            im_resized = np.uint8(cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                                    interpolation=cv2.INTER_LINEAR))

            im_resized = cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB)

            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])

            im_resized = np.array(transform(im_resized).permute(1, 2, 0))

            ims.append(im_resized)
            im_scales.append(im_scale)
    else:
        print("prep_im_for_blob error!!")
        raise AssertionError

    return ims, im_scales


def get_im_blob_sizes(im_shape, target_sizes, max_size):
    """Calculate im blob size for multiple target_sizes given original im shape
    """
    im_size_min = np.min(im_shape)
    im_size_max = np.max(im_shape)
    im_sizes = []
    for target_size in target_sizes:
        im_scale = get_target_scale(im_size_min, im_size_max, target_size, max_size)
        im_sizes.append(np.round(im_shape * im_scale))
    return np.array(im_sizes)


def get_target_scale(im_size_min, im_size_max, target_size, max_size):
    """Calculate target resize scale
    """
    im_scale = float(target_size) / float(im_size_max)
    # Prevent the biggest axis from being more than max_size
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    return im_scale


def zeros(shape, int32=False):
    """Return a blob of all zeros of the given shape with the correct float or
    int data type.
    """
    return np.zeros(shape, dtype=np.int32 if int32 else np.float32)


def ones(shape, int32=False):
    """Return a blob of all ones of the given shape with the correct float or
    int data type.
    """
    return np.ones(shape, dtype=np.int32 if int32 else np.float32)


def serialize(obj):
    """Serialize a Python object using pickle and encode it as an array of
    float32 values so that it can be feed into the workspace. See deserialize().
    """
    return np.fromstring(pickle.dumps(obj), dtype=np.uint8).astype(np.float32)


def deserialize(arr):
    """Unserialize a Python object from an array of float32 values fetched from
    a workspace. See serialize().
    """
    return pickle.loads(arr.astype(np.uint8).tobytes())
