#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=iql_train
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=./logs/out_%j_%x_%N.log  # includes time stamp (t), job ID(j), job name (x), and node name (N)
#SBATCH --error=./logs/err_%j_%x_%N.err



module --force purge
module load python-pytorch
source /scratch/project_2009050/rl/bin/activate
cd /projappl/project_2009050/c3jepa-wm/mains/baselines/og-marl

# export WANDB_START_METHOD=thread
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# srun torchrun --standalone --nnodes=1 --nproc_per_node=1 train_wm.py --config ../cfgs/findgoal/mawm/ablations/datasize/mawm_ds_1k.yaml --env_file ../.env --timestamp ${ts}
srun python train.py
