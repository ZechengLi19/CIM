# Editable
cfg_file=./configs/resnet50_voc.yaml
output_file=./Outputs/resnet50_voc/Mar22-00-46-22_user-Super-Server_step
iter_time=model_step44999
dataset=voc2012trainaug

############
# Not editable
ckpt=${output_file}/ckpt/${iter_time}.pth
result_pkl=${output_file}/trainaug/${iter_time}/discovery.pkl
output_dir=${output_file}/trainaug/${iter_time}

# generate discovery.pkl on training set
python ./tools/test_net.py \
--cfg ${cfg_file} \
--load_ckpt ${ckpt} \
--dataset ${dataset}

# generate pseudo labels (coco format)
python ./tools/generate_mask_for_MaskRCNN.py \
--cfg ${cfg_file} \
--result_path ${result_pkl} \
--dataset ${dataset}

# filter out low confidence pseudo labels
python ./tools/change_pesudo_label_thr.py \
--output_dir ${output_dir} \
--thr 0.3
#############