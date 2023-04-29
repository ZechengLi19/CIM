echo $(nvidia-smi)
source activate wsis_py_3.6

cfg_file=./configs/re_baseline/resnet50_cobsbd_refine_mode_mask_fuse.yaml

output_file=Outputs/resnet50_cobsbd_refine_mode_mask_fuse/Mar22-00-46-22_user-Super-Server_step

dataset=voc2012sbdval
#dataset=coco2017val
#dataset=coco2017test
#dataset=voc2012trainaug

iter_time=model_step44999
ckpt=${output_file}/ckpt/${iter_time}.pth
result_pkl=${output_file}/test/${iter_time}/detections.pkl

#CUDA_VISIBLE_DEVICES=1,2,3 python -u tools/test_net.py --cfg ${cfg_file} \
#--load_ckpt ${ckpt} \
#--dataset ${dataset} \

CUDA_VISIBLE_DEVICES=1,2,3 python -u tools/my_eval_mask_coco_multi.py --cfg ${cfg_file} \
--result_path ${result_pkl} \
--dataset ${dataset}

echo "Done"
