import argparse
from PIL import Image
import numpy as np

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
    imnew = im.resize(size, resample=func[interp])
    return np.array(imnew)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', required=True, choices=["voc", "coco"],
        help='Dataset to use')
    return parser.parse_args()