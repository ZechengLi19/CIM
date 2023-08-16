# Complete Instances Mining for Weakly Supervised Instance Segmentation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/complete-instances-mining-for-weakly/image-level-supervised-instance-segmentation)](https://paperswithcode.com/sota/image-level-supervised-instance-segmentation?p=complete-instances-mining-for-weakly)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/complete-instances-mining-for-weakly/image-level-supervised-instance-segmentation-2)](https://paperswithcode.com/sota/image-level-supervised-instance-segmentation-2?p=complete-instances-mining-for-weakly)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/complete-instances-mining-for-weakly/image-level-supervised-instance-segmentation-1)](https://paperswithcode.com/sota/image-level-supervised-instance-segmentation-1?p=complete-instances-mining-for-weakly)

This project hosts the code for implementing the CIM algorithm for weakly supervised instance segmentation.
![CIM](docs/pipeline.png)

## Installation
Please follow the instructions in [INSTALL.md](./docs/INSTALL.md).

## Preparation
Please follow the instructions in [DATASET.md](./docs/DATASET.md).

## Experiments
### Training
```bash
bash ./scripts/train_CIM.sh
```

### Evaluation
```bash
bash ./scripts/eval_CIM.sh
```

### Mask R-CNN Refinement
```bash
# generate pseudo labels from CIM for training Mask R-CNN
bash ./scripts/generate_msrcnn_label.sh
```
Then, we use mmdetection for Mask R-CNN Refinement.

### Visualization
```bash
bash ./scripts/visual_result_mmcv.sh
```

## Results
Results of instance segmentation on the VOC2012 and COCO datasets can be downloaded from [here](https://1drv.ms/f/s!Ah9g93YHHTrAaje14InpZd_XDEw?e=xhEhxT).

## Contact
If you have any questions, please feel free to contact Zecheng Li (lizecheng19@gmail.com). Thank you.

## Acknowledgement
Our implementation is based on these repositories:
- (PRM) https://github.com/ZhouYanzhao/PRM
- (PCL) https://github.com/ppengtang/pcl.pytorch
- (HRNet) https://github.com/HRNet/HRNet-Image-Classification
- (mmdetection) https://github.com/open-mmlab/mmdetection

## Citation
If you use CIM or this repository in your work, please cite:
```
@inproceedings{zecheng2023CIM,
  title={Complete Instances Mining for Weakly Supervised Instance Segmentation},
  author={Li, Zecheng and Zeng, Zening and Liang, Yuqi and Yu, Jin-Gang},
  booktitle={International Joint Conference on Artificial Intelligence},
  year={2023},
}
```