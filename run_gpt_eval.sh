#!/bin/bash
#SBATCH --job-name="test_rnadom_cpu"
#SBATCH --output="test.out.%j.%N.out"
#SBATCH --partition=cpu
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=1   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --account=bebr-delta-cpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH -t 24:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --mail-user=mkhan@wpi.edu    # <- replace "NetID" with your University NetID
#SBATCH --mail-type=BEGIN,END

source venv-gemma/bin/activate
# Exit immediately if a command exits with a non-zero status
set -e

python3 gpt_evaluation.py