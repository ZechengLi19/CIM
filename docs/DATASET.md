### Datasets
We use the following datasets in our experiments:
- [**VOC2012**](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- [**COCO2017**](https://cocodataset.org/#home)

### Dataset structure
For simplicity, our datasets are structured in the following way:
```
/CIM/data/
├── VOC2012/
│   ├── annotations/
│   ├── JPEGImages/
│   ├── COB_SBD_trainaug/
│   ├── COB_SBD_val/
│   └── Center_points/
│
├── coco2017/
│   ├── annotations/
│   ├── train2017/
│   ├── val2017/
│   ├── test2017/
│   └── COB-COCO/
│
├── model_weight/
│   ├── prm_voc.pth
│   ├── prm_coco.pth
│   ├── vgg16_caffe.pth
│   └── hrnetv2_w48_imagenet_pretrained.pth
│ 
├── label_assign/
│   ├── voc_2012_label_assign.pkl
│   └── coco_2017_label_assign.pkl
│
├── cob/
│   ├── voc_2012_trainaug.pkl
│   ├── voc_2012_val.pkl
│   ├── coco_2017_train.pkl
│   ├── coco_2017_val.pkl
│   └── coco_2017_test.pkl
│
├── cob_asy_iou/
│   ├── VOC2012/
│   └── coco2017/
│
└── cob_iou/
    ├── VOC2012/
    └── coco2017/

```

#### Note: 
- **`VOC2012/annotations/`** is a folder containing label files in json format. You can convert .xml format annotations to coco format, or you can use our pre-processed annotations [here](https://drive.google.com/drive/folders/1CrzR3V5dgrwzaQYER5AwnLD_OfpjUylh?usp=drive_link).
- **`VOC2012/COB_SBD_trainaug/`**, **`VOC2012/COB_SBD_val/`** and **`coco2017/COB-COCO/`** are folders containing COB files. You can download **`VOC2012/COB_SBD_trainaug/`**, **`VOC2012/COB_SBD_val/`** from [here](https://drive.google.com/drive/folders/16Nvm3AMq3JFpOSIznUpZhVI0QrJazEmw?usp=sharing).**`coco2017/COB-COCO/`** can be downloaded from [here](https://rec.ustc.edu.cn/share/7dcb25c0-c8f1-11ee-b716-33a1b4b1b28b).
- **`VOC2012/Center_points/`** is a folder containing point-level label files. We download them from [BESTIE](https://drive.google.com/file/d/1Xg_F8MjIOG4w5f_L9-K1dEh8Wh7mpe_2/view?usp=drive_link).
- **`model_weight/`**  is a folder containing weight of models. Weights can be downloaded from [here](https://drive.google.com/drive/folders/1kzFsaPlbYK0OY31a7vqsRLDaJQ2BbAs0?usp=sharing). hrnetv2_w48_imagenet_pretrained.pth can be downloaded from [here](https://github.com/HRNet/HRNet-Image-Classification).
- **`label_assign/`** contains pre-computed pseudo labels for VOC2012 and COCO2017 datasets, link is [here](https://drive.google.com/drive/folders/1j44PAimT7v4RkkOlKbbqcCLAiNf9sXjN?usp=sharing). It also can be created by running **`python tools/pre/AGPL_label_assign.py`**.
- **`cob/`** is a folder containing two dataset proposals. The pkl files contain proposals that are scaled to a size of 7*7. These files will be used in RoiAlign operation. They can be downloaded from [here](https://drive.google.com/drive/folders/144_iTb57xnvBL8R7eDm2U_WF1UBQCtYz?usp=sharing). They also can be created by running **`python tools/pre/generate_7_7_voc.py`** and **`python tools/pre/generate_7_7_coco.py`**. 
- **`cob_iou/`** can be downloaded from [here](https://drive.google.com/drive/folders/1BwS_FaM9OOWzpjAR5Tul2gLgFbv0iN9X?usp=sharing). It also can be created by running **`python tools/pre/create_cob_iou.py`**.
- **`cob_asy_iou/`** can be downloaded from [here](https://drive.google.com/drive/folders/1PZfP9Wz0uL33wMcY6wX--C6Wb_cH1ZHT?usp=sharing). It also can be created by running **`python tools/pre/create_cob_asy_iou.py`**.
