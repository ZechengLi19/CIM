# Editable
dataset=coco_val
result_file=result/CIM-COCO-val.json
save_dir=./vis_COCO_val

##############
# Not editable
# train CIM
python ./visualize/vis_json_mmcv.py --dataset ${dataset} \
--result_file ${result_file} \
--save_dir ${save_dir}
############