# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import torch
import torchvision.transforms.functional as F
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.voc import download_extract
from PIL import Image
import xml.etree.ElementTree as ET
import os
import collections
from torchvision import transforms

VOC_CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
CLS_TO_IND = {k: v for v, k in enumerate(VOC_CLASSES)}
  
## between comments taken from the torchvision source code with modifications to include
DATASET_YEAR_DICT = {
    '2012': { 
        'trainval': {
            'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
            'filename': 'VOCtrainval_11-May-2012.tar',
            'md5': '6cd6e144f989b92b3379bac3b3de84fd',
            'base_dir': 'VOCdevkit/VOC2012'
        },
        'test': {
            'url': 'http://pjreddie.com/media/files/VOC2012test.tar',
            'filename': 'VOC2012test.tar',
            'md5': '',
            'base_dir': 'VOCdevkit/VOC2012'
        }
    },
    '2011': {
        'trainval': {
            'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
            'filename': 'VOCtrainval_25-May-2011.tar',
            'md5': '6c3384ef61512963050cb5d687e5bf1e',
            'base_dir': 'TrainVal/VOCdevkit/VOC2011'
        }
    },
    '2010': {
        'trainval': {
            'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
            'filename': 'VOCtrainval_03-May-2010.tar',
            'md5': 'da459979d0c395079b5c75ee67908abb',
            'base_dir': 'VOCdevkit/VOC2010'
        }
    },
    '2009': {
        'trainval': {
            'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
            'filename': 'VOCtrainval_11-May-2009.tar',
            'md5': '59065e4b188729180974ef6572f6a212',
            'base_dir': 'VOCdevkit/VOC2009'
        }
    },
    '2008': {
        'trainval': {
            'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
            'filename': 'VOCtrainval_11-May-2012.tar',
            'md5': '2629fa636546599198acfcfbfcf1904a',
            'base_dir': 'VOCdevkit/VOC2008'
        }
    },
    '2007': {
        'trainval': {
            'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
            'filename': 'VOCtrainval_06-Nov-2007.tar',
            'md5': 'c52e279531787c972589f7e41ab4ae64',
            'base_dir': 'VOCdevkit/VOC2007'
        },
        'test': {
            'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
            'filename': 'VOCtest_06-Nov-2007.tar',
            'md5': 'b6e924de25625d8de591ea690078ad9f',
            'base_dir': 'VOCdevkit/VOC2007'
        }
    }
}

class VOCDetection(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root,
                 image_set='sbdval',
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(VOCDetection, self).__init__(root, transforms, transform, target_transform)
        self.root = root 
        self.image_set = image_set
        base_dir = 'VOCdevkit/VOC2012'
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val" or a valid'
                'image_set from the VOC ImageSets/Main folder.')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
## End code from elsewhere
open_transform = transforms.Compose(
        [transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )


class VOCWeak(VOCDetection):
    def __init__(self, 
                 root,
                 image_set='sbdval'):
        super(VOCWeak, self).__init__(root, image_set)
    
    def __getitem__(self, index):
        # img = F.to_tensor(Image.open(self.images[index]).convert('RGB'))
        img =  open_transform(Image.open(self.images[index]).convert('RGB'))
        tree = ET.parse(self.annotations[index])

        objects = tree.findall('object')
        num_objs = len(objects)
        boxes = torch.zeros((num_objs, 4))
        boxes_cl = torch.zeros((num_objs,)).long()
    
        for i, ob in enumerate(objects):
            bbox = ob.find('bndbox')
            boxes[i, :] = torch.tensor([float(bbox.find('xmin').text),
                                        float(bbox.find('ymin').text),
                                        float(bbox.find('xmax').text),
                                        float(bbox.find('ymax').text)])
            boxes_cl[i] = CLS_TO_IND[ob.find('name').text.lower().strip()]

        img_labels = torch.zeros((21,))
        img_labels[boxes_cl] = 1

        return img, img_labels, boxes, boxes_cl, tree.find('filename').text[:-4]
