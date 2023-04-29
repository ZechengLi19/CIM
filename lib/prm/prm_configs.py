# -*- coding: utf-8 -*-

import torch
from torchvision import transforms 
from torch.utils.data import Dataset

import os
from PIL import Image
import scipy.io
import numpy as np


train_transform = transforms.Compose(
        [transforms.Resize([448, 448]),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

open_transform = transforms.Compose(
        [transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

categories_dict = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3,
    'bottle':4, 'bus':5, 'car':6, 'cat':7, 'chair':8,
    'cow':9, 'diningtable':10, 'dog':11, 'horse':12,
    'motorbike':13, 'person':14, 'pottedplant':15,
    'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}    

def ismember(a_vec,b_vec):
    '''
    matlab function: ismember
    https://stackoverflow.com/questions/15864082/python-equivalent-of-matlabs-ismember-function/15864429
    '''
    bool_ind = np.isin(a_vec,b_vec)
    common = a_vec[bool_ind]
    common_unique, common_inv  = np.unique(common, return_inverse=True)     # common = common_unique[common_inv]
    b_unique, b_ind = np.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]

    #common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]
    return bool_ind #, common_ind[common_inv]


from fnmatch import fnmatch    
def finetune(model, base_lr=0.01, groups={'feature':0.01}, ignore_the_rest = False, raw_query= False) : 
    parameters = [dict(params=[], names=[], query=query if raw_query else '*'+query+'*', lr=lr*base_lr) for query, lr in groups.items()]
    rest_parameters = dict(params=[], names=[], lr=base_lr)
    for k, v in model.named_parameters():
        for group in parameters:  
            if fnmatch(k, group['query']): # group['query'] = '*feature*'
                group['params'].append(v)
                group['names'].append(k)
            else:
                rest_parameters['params'].append(v)
                rest_parameters['names'].append(k)
    if not ignore_the_rest:  # True
        parameters.append(rest_parameters)
    for group in parameters:
        group['params'] = iter(group['params'])
    return parameters    
    
    
cls_labels_dict = np.load('lib/prm/cls_labels.npy', allow_pickle=True).item()
def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.int32)
    return img_name_list
def load_image_label_list_from_npy(img_name_list):
    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])
def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]
#
class VOC_Classification(Dataset):
    def __init__(self, data_dir = 'data/VOC2012',  #/mnt/jaring/VOCdevkit',
                 split = 'trainaug',  transform = train_transform, target_transform = None ):

        self.data_dir = data_dir
        self.split = split
        self.image_dir = os.path.join(self.data_dir, 'JPEGImages')

        self.img_name_list =  load_img_name_list(os.path.join( self.data_dir, 'ImageSets/Main', split + '.txt') )
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        im_name = self.img_name_list[index]
        filename = decode_int_filename(im_name)
        img = Image.open(os.path.join(self.image_dir, filename + '.jpg')).convert('RGB')
        target = torch.from_numpy(self.label_list[index])

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.img_name_list)
    
    
    