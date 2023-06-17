echo $(nvidia-smi)
source activate wsis_py_3.6

python ./visualize/vis_json_mmcv.py --dataset coco_val \
--result_file /home/data2/lzc/WSIS-Benchmark/code/result_file/json_file/CIM_result/result/CIM-COCO-val.json \
--save_dir /home/data2/lzc/WSIS-Benchmark/code/result_file/json_file/CIM_result/COCO_val \

echo "Done"