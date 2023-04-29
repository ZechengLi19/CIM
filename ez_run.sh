echo $(nvidia-smi)
source activate wsis_py_3.6

#cfg_file=./configs/baseline/resnet50_cobcoco2017_refine_mode_mask_fuse.yaml
#
#output_file=Outputs/resnet50_cobcoco2017_refine_mode_mask_fuse/Nov28-20-17-47_user-Super-Server_step
##dataset=voc2012sbdval
#dataset=coco2017val
##dataset=coco2017test
##dataset=voc2012trainaug
#
#iter_time=model_step239999
#ckpt=${output_file}/ckpt/${iter_time}.pth
#result_pkl=${output_file}/test/${iter_time}/detections.pkl
#
#CUDA_VISIBLE_DEVICES=3 python -u tools/test_net.py --cfg ${cfg_file} \
#--load_ckpt ${ckpt} \
#--dataset ${dataset} \
#
#CUDA_VISIBLE_DEVICES=3 python -u tools/my_eval_mask_coco_multi.py --cfg ${cfg_file} \
#--result_path ${result_pkl} \
#--dataset ${dataset}

python -u tools/prm_tools/prm_labelassign_refine_mode_json.py

echo "Done"

