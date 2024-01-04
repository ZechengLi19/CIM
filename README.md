# Complete Instances Mining for Weakly Supervised Instance Segmentation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/complete-instances-mining-for-weakly/image-level-supervised-instance-segmentation)](https://paperswithcode.com/sota/image-level-supervised-instance-segmentation?p=complete-instances-mining-for-weakly)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/complete-instances-mining-for-weakly/image-level-supervised-instance-segmentation-2)](https://paperswithcode.com/sota/image-level-supervised-instance-segmentation-2?p=complete-instances-mining-for-weakly)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/complete-instances-mining-for-weakly/image-level-supervised-instance-segmentation-1)](https://paperswithcode.com/sota/image-level-supervised-instance-segmentation-1?p=complete-instances-mining-for-weakly)

This project hosts the code for implementing the CIM algorithm for weakly supervised instance segmentation.
![CIM](docs/pipeline.png)

## Quick View
Since running the code requires preparing a lot of data, if you just want to understand how we implement CIM, you can directly choose to read the paper and the following code.
- **Paper** [[Paper](https://www.ijcai.org/proceedings/2023/0127.pdf)]
- **Pipeline** [[code](./lib/modeling/model_builder.py)]
- **CIM Strategy** [[code](./lib/modeling/heads.py)]

## Installation
Please follow the instructions in [INSTALL.md](./docs/INSTALL.md).

## Preparation
Please follow the instructions in [DATASET.md](./docs/DATASET.md).

## Experiments
Before starting the experiment. You need to update these four values ```cfg_file, output_file, dataset, iter_time``` according to your needs. 
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
### VOC2012
| Method |     Backbone      | mAP25 | mAP50 | mAP70 |  mAP75   |
|:------:|:-----------------:|:-----:|:-----:|:-----:|:--------:|
|  CIM   |     ResNet-50     | 64.9 | 51.1  | 32.4  |  26.1    |
|  CIM   |      VGG-16       | 65.6 | 50.8  | 31.0  |     25.2     |
|  CIM   |    HRNet-W48      | 68.3 | 52.6  | 33.7  |     28.4     |
|  CIM+  |   ResNet-50    | 68.7 | 55.9  | 37.1  |      30.9    |

### COCO val2017
| Method |     Backbone      |  AP  | mAP50 | mAP75 |
|:------:|:-----------------:|:----:|:-----:|:-----:|
|  CIM   |     ResNet-50     | 11.9 | 22.8  | 11.1  | 
|  CIM+  |   ResNet-50    | 17.0 |   29.4    | 17.0  | 

### COCO test-dev
| Method |     Backbone      |  AP  | mAP50 | mAP75 |
|:------:|:-----------------:|:----:|:-----:|:-----:|
|  CIM   |     ResNet-50     | 12.0 | 23.0  |   11.3    | 
|  CIM+  |   ResNet-50    | 17.2 |   29.7     | 17.3  | 
### Download
Results of instance segmentation on the VOC2012 and COCO datasets can be downloaded from [OneDrive](https://1drv.ms/f/s!Ah9g93YHHTrAaje14InpZd_XDEw?e=xhEhxT) | [Google Drive](https://drive.google.com/drive/folders/11DrIJmIy7j7rIrUlvGNLJKFUnKbUjdWc?usp=sharing).

## Contact
If you have any questions, please feel free to contact Zecheng Li (lizecheng19@gmail.com). Thank you.

## Acknowledgement
Our implementation is based on these repositories:
- (PRM) https://github.com/ZhouYanzhao/PRM
- (PCL) https://github.com/ppengtang/pcl.pytorch
- (MIST) https://github.com/NVlabs/wetectron
- (HRNet) https://github.com/HRNet/HRNet-Image-Classification
- (mmdetection) https://github.com/open-mmlab/mmdetection

## Citation
If you find this work useful, please consider giving it a star ⭐ and citing our paper in your work:
```
@inproceedings{zecheng2023CIM,
  title={Complete Instances Mining for Weakly Supervised Instance Segmentation},
  author={Li, Zecheng and Zeng, Zening and Liang, Yuqi and Yu, Jin-Gang},
  booktitle={International Joint Conference on Artificial Intelligence},
  year={2023},
}
```