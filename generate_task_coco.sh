#!/bin/bash
#SBATCH -J generate_task_coco
#SBATCH --output=./log/generate_task_coco_out.log
#SBATCH --error=./log/generate_task_coco_error.log
#SBATCH -p cpu
#SBATCH -w gpu[04]
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16


echo "SLURM_JOB_PARTITION={$SLURM_JOB_PARTITION}"
echo "SLURM_JOB_NODELIST={$SLURM_JOB_NODELIST}"
echo $(nvidia-smi)
source activate wsrcnn
srun python ./tools/pre/generate_proposalPKL_coco2017.py
echo "Done"

