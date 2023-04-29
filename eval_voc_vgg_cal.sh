echo $(nvidia-smi)
source activate wsis_py_3.6

cfg_file=./configs/baselines/vgg16_cobsbd_refine_mode_pamr_mask_fuse.yaml
branch=0
iter_time=2499
output_file=vgg16_cobsbd_refine_mode_pamr_mask_fuse/Nov15-18-14-26_user-Super-Server_step
dataset=voc2012sbdval
#dataset=coco2017val
#dataset=voc2012trainaug

ckpt=Outputs/${output_file}/ckpt/model_step${iter_time}.pth
result_pkl=Outputs/${output_file}/test/model_step${iter_time}/detections.pkl

CUDA_VISIBLE_DEVICES=2,3 python -u tools/test_net.py --cfg ${cfg_file} \
--load_ckpt ${ckpt} \
--dataset ${dataset} \
--set after_diffuse True \
refine_model_cal True \
easy_case_mining False \
cal_refine_branch ${branch}

CUDA_VISIBLE_DEVICES=2,3 python -u tools/my_eval_mask_coco_multi.py --cfg ${cfg_file} \
--result_path ${result_pkl} \
--dataset ${dataset}

CUDA_VISIBLE_DEVICES=2,3 python -u tools/test_net.py --cfg ${cfg_file} \
--load_ckpt ${ckpt} \
--dataset ${dataset} \
--set after_diffuse False \
refine_model_cal True \
easy_case_mining False \
cal_refine_branch ${branch}

CUDA_VISIBLE_DEVICES=2,3 python -u tools/my_eval_mask_coco_multi.py --cfg ${cfg_file} \
--result_path ${result_pkl} \
--dataset ${dataset}

echo "Done"
