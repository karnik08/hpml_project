#!/bin/bash
#
#SBATCH --job-name=hpml_project_4gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=05:00:00
#SBATCH --mem=320GB
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --output=./result_4gpu.out
#SBATCH --error=./result_4gpu.err

singularity exec --nv --overlay $SCRATCH/singular/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c " source /ext3/env.sh;python3 main.py" 

