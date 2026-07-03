#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=wm_eval
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=./logs/out_%j_%x_%N.log  # includes time stamp (t), job ID(j), job name (x), and node name (N)
#SBATCH --error=./logs/err_%j_%x_%N.err



module --force purge
module load pytorch
source /scratch/project_2009050/rl/bin/activate
cd /projappl/project_2009050/c3jepa-wm/mains/

# export WANDB_START_METHOD=thread
# ts=$(date +%Y%m%d_%H%M%S)
# srun torchrun --standalone --nnodes=1 --nproc_per_node=1 train_wm.py --config ../cfgs/findgoal/mawm/ablations/datasize/mawm_ds_1k.yaml --env_file ../.env --timestamp ${ts}
srun python eval.py


# singularity exec --nv \
#   -B /lib/x86_64-linux-gnu:/host_glibc \
#   --env LD_LIBRARY_PATH=/host_glibc:$LD_LIBRARY_PATH \
#   -B $SCRATCH:$SCRATCH \
#   -e ACCEPT_EULA=Y \
#   isaac-sim_4.5.0.sif \
#   /isaac-sim/isaac-sim.sh