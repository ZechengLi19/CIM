# Complete Instances Mining for WSIS

## Framework

![CIM](docs/pipeline.png) 

## Code
Code is cleaning. 

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
