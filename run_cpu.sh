#!/bin/bash
#SBATCH --job-name="train_data_check"
#SBATCH --output="test.out.%j.%N.out"
#SBATCH --partition=gpuA40x4,gpuA100x8
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=1   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=1
#SBATCH --account=bebr-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH -t 5:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out


source venv-gemma/bin/activate
# Exit immediately if a command exits with a non-zero status
set -e

python3 dataset_LB.py