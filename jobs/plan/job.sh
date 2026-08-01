#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=wm_eval
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=./logs/out_%j_%x_%N.log  # includes time stamp (t), job ID(j), job name (x), and node name (N)
#SBATCH --error=./logs/err_%j_%x_%N.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

export HYDRA_FULL_ERROR=1

module --force purge
module load python-pytorch
source /scratch/project_2009050/rl/bin/activate
cd /projappl/project_2009050/c3jepa-wm/mains/

srun python eval.py
