# Editable
cfg_file=./configs/baseline/resnet50_voc.yaml
dataset=voc2012trainaug

##############
# Editable
# train CIM
python ./tools/train.py \
--dataset ${dataset} \
--cfg ${cfg_file}
############