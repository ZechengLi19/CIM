echo $(nvidia-smi)
source activate wsis_py_3.6

#cfg_file=./configs/baseline/resnet50_cobcoco2017_refine_mode_mask_fuse.yaml
#
#output_file=Outputs/resnet50_cobcoco2017_refine_mode_mask_fuse/Nov22-11-43-10_user-Super-Server_step
#iter_time=model_step239999

cfg_file=./configs/Final_baseline/hrnet48_cobsbd_refine_mode_mask_fuse.yaml

output_file=Outputs/hrnet48_cobsbd_refine_mode_mask_fuse/Nov21-23-46-19_user-Super-Server_step
iter_time=model_step44999

dataset=voc2012trainaug
#dataset=coco2017train

ckpt=${output_file}/ckpt/${iter_time}.pth
result_pkl=${output_file}/trainaug/${iter_time}/discovery.pkl

org_dir=${output_file}/trainaug/${iter_time}

CUDA_VISIBLE_DEVICES=2,3 python -u tools/test_net.py --cfg ${cfg_file} \
--load_ckpt ${ckpt} \
--dataset ${dataset} \

CUDA_VISIBLE_DEVICES=2,3 python -u ./tools/coco_mrcnn_multi.py --cfg ${cfg_file} \
--result_path ${result_pkl} --dataset ${dataset}

python -u ./tools/change_pesudo_label_thr.py \
--org_dir ${org_dir} --thr 0.3

echo "Done"

