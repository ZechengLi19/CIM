echo $(nvidia-smi)
source activate wsis_py_3.6
srun python ./tools/prm_tools/prm_labelassign.py

echo "Done"
