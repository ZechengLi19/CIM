# Complete Instances Mining for WSIS

## Framework

![CIM](docs/pipeline.png) 

## Code
- The code is being cleaned up.

I'm currently working on a graduation project and may be slower to update.
However, if you have any questions, please feel free to contact me (lizecheng19@gmail.com). 

## Installation
### Setup with Conda
We suggest to create a new conda environment and install all the relevant dependencies. 

```bash
# create environment
conda create --name CIM python=3.6
conda activate CIM

# Install requirements
pip install -r requirements.txt
```

## Preparation
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
│   └── output/
│
├── coco2017/
│   ├── annotations/
│   ├── train2017/
│   ├── val2017/
│   ├── test2017/
│   └── COB-COCO/
│
├── label_assign/
│   ├── voc_2012_label_assign.pkl
│   └── coco_2017_label_assign.pkl
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
- VOC2012/annotations is a folder containing label files in json format.
- VOC2012/COB_SBD_trainaug and VOC2012/output are folders containing COB files.
- coco2017/COB-COCO is a folder containing COB files.
- label_assign/ contains pre-computed pseudo labels for VOC2012 and COCO2017 datasets, link is [here](https://drive.google.com/drive/folders/1j44PAimT7v4RkkOlKbbqcCLAiNf9sXjN?usp=sharing). It also can be created by running **`tools\prm_tools\prm_label_assign.py`**, the weights of prm model can be downloaded from [here](https://drive.google.com/drive/folders/1kzFsaPlbYK0OY31a7vqsRLDaJQ2BbAs0?usp=sharing).
- cob_iou/ can be downloaded from [here](https://drive.google.com/drive/folders/1BwS_FaM9OOWzpjAR5Tul2gLgFbv0iN9X?usp=sharing). It also can be created by running **`tools\prm_tools\create_cob_iou.py`**.
- cob_asy_iou/ can be downloaded from [here](https://drive.google.com/drive/folders/1PZfP9Wz0uL33wMcY6wX--C6Wb_cH1ZHT?usp=sharing). It also can be created by running **`tools\prm_tools\create_cob_asy_iou.py`**.

## Experiments
### Training
We use the following script to train CIM model.

For VOC2012 dataset: 
```bash
python -u ./tools/train_net_step.py \
--dataset voc2012trainaug \
--cfg ./configs/baseline/resnet50_voc.yaml
```

For COCO2017 dataset:
```bash
python -u ./tools/train_net_step.py \
--dataset coco2017 \
--cfg ./configs/baseline/resnet50_coco2017.yaml
```

### Evaluation

We use the following script to evaluate CIM model.

For VOC2012 dataset: 
```bash
cfg_file=./configs/baseline/resnet50_voc.yaml
output_file=Outputs/resnet50_voc/Mar22-00-46-22_user-Super-Server_step
dataset=voc2012sbdval
iter_time=model_step44999

ckpt=${output_file}/ckpt/${iter_time}.pth
result_pkl=${output_file}/test/${iter_time}/detections.pkl

python -u tools/my_eval_mask_coco_multi.py \
--cfg ${cfg_file} \
--result_path ${result_pkl} \
--dataset ${dataset}
```

For COCO2017 dataset: 
```bash
cfg_file=./configs/baseline/resnet50_coco2017.yaml
output_file=Outputs/resnet50_coco2017/Mar15-18-14-26_user-Super-Server_step
dataset=coco2017val
iter_time=model_step239999

ckpt=${output_file}/ckpt/${iter_time}.pth
result_pkl=${output_file}/test/${iter_time}/detections.pkl

python -u tools/my_eval_mask_coco_multi.py \
--cfg ${cfg_file} \
--result_path ${result_pkl} \
--dataset ${dataset}
```

## Results
Results of instance segmentation on the VOC2012 and COCO datasets can be downloaded [here](https://drive.google.com/file/d/14TuME6jLEMdlD6HUMSLHDv09oMwE0K_3/view?usp=share_link).

## Acknowledgement
Our implementation is based on these repositories:
- (PCL) https://github.com/ppengtang/pcl.pytorch
- (mmdetection) https://github.com/open-mmlab/mmdetection
