#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=data_collection
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=8:00:00
#SBATCH --output=./logs/out_%j_%x_%N.log  # includes time stamp (t), job ID(j), job name (x), and node name (N)
#SBATCH --error=./logs/err_%j_%x_%N.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module --force purge
module load pytorch
source /scratch/project_2009050/rl/bin/activate
cd /projappl/project_2009050/c3jepa-wm/mains/

srun python collect.py