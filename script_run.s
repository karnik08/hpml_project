#!/bin/bash
#
#SBATCH --job-name=hpml_project_256
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --output=./result_256.out
#SBATCH --error=./result_256.err

singularity exec --nv --overlay $SCRATCH/singular/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c " source /ext3/env.sh;python3 main.py" 

