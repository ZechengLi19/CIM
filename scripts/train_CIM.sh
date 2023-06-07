# Editable
cfg_file=./configs/resnet50_voc.yaml
dataset=voc2012trainaug

##############
# Editable
# train CIM
CUDA_VISIBLE_DEVICES=0 python ./tools/train.py \
--dataset ${dataset} \
--cfg ${cfg_file}
############