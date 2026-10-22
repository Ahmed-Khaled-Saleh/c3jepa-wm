#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=wm_finetune
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=./logs/out_%j_%x_%N.log  # includes time stamp (t), job ID(j), job name (x), and node name (N)
#SBATCH --error=./logs/err_%j_%x_%N.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}


module --force purge
module load python-pytorch/2.10
source /scratch/project_2009050/rl/bin/activate
cd /projappl/project_2009050/c3jepa-wm/mains/

# JEPA: the blind model of jepa.sh (else get_ckp_path's for the host); VQVAE likewise
srun python train.py pipeline=wm_finetune logging_params.project_name="phase-2-wm-finetune" ${JEPA:+pipeline.model.jepa.checkpoint_path=$JEPA} ${VQVAE:+pipeline.model.vqvae.checkpoint_path=$VQVAE} #="jepa_train"