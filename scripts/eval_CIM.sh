# Editable
cfg_file=./configs/baseline/resnet50_voc.yaml
output_file=./Outputs/resnet50_voc/Mar22-00-46-22_user-Super-Server_step
dataset=voc2012sbdval
iter_time=model_step44999

##############
# Not editable
ckpt=${output_file}/ckpt/${iter_time}.pth
result_pkl=${output_file}/test/${iter_time}/detections.pkl

# generate detections.pkl on test set
python -u tools/test_net.py \
--cfg ${cfg_file} \
--load_ckpt ${ckpt} \
--dataset ${dataset} \

# report mAP
python tools/evaluation.py \
--cfg ${cfg_file} \
--result_path ${result_pkl} \
--dataset ${dataset}
###########
