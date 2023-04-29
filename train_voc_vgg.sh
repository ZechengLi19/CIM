echo $(nvidia-smi)
source activate wsis_py_3.6

CUDA_VISIBLE_DEVICES=0 python -u ./tools/train_net_step.py --dataset voc2012trainaug \
--cfg ./configs/baseline/resnet50_cobsbd_refine_mode_mask_fuse.yaml \

echo "Done"
