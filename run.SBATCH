#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:v100:1  ## To request specific GPU (v100 or rtx8000)
#SBATCH --job-name=torch

module purge

singularity exec --nv \
	    --overlay /scratch/sja491/server1/overlay-7.5GB-300K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "python3 main.py"
