#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=deploy
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=./logs/out_%j_%x_%N.log  # includes job ID (j), job name (x) and node name (N)
#SBATCH --error=./logs/err_%j_%x_%N.err

# Deployment (mains/deploy.py, configs/deploy/config.yaml): the agents plan with the world model on the test layouts
# while their messages go through comm-core's link (ray-traced channel, Rician fading, C3-WM's power control, BPSK, CRC);
# per method the success rate and, over the link, its reliability and power. The GPU plans, and traces the pairs of
# cells not traced yet (Sionna RT). Override anything at submission, e.g.
#   JEPA=... VQVAE=... sbatch deploy.sh
#   JEPA=... VQVAE=... EXTRA="methods=[learned] episodes=120 planner.num_samples=256" sbatch deploy.sh
#   JEPA=... VQVAE=... METHODS="[random,perfect channel]" EXTRA="episodes=40 planner.cost=terminal" sbatch deploy.sh
# METHODS is passed as one argument (method names have spaces, which EXTRA's words would split).

JEPA=${JEPA:?set JEPA to the fine-tuned world model checkpoint}
VQVAE=${VQVAE:?set VQVAE to the VQ-VAE checkpoint}
WANDB_MODE=${WANDB_MODE:-online}
EXTRA=${EXTRA:-}
METHODS=${METHODS:-}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MPLBACKEND=Agg

module --force purge
module load python-pytorch/2.10
source /scratch/project_2009050/rl/bin/activate
cd /projappl/project_2009050/c3jepa-wm/mains/
mkdir -p logs

echo "------------------------------------------------"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME: jepa=$JEPA vqvae=$VQVAE methods=$METHODS $EXTRA"
nvidia-smi -L
python -c "import drjit as dr, mitsuba as mi; assert dr.has_backend(dr.JitBackend.CUDA), 'Dr.Jit sees no CUDA device'; mi.set_variant('cuda_ad_mono_polarized'); import sionna.rt" || exit 1
echo "------------------------------------------------"

srun python deploy.py checkpoints.jepa="$JEPA" checkpoints.vqvae="$VQVAE" wandb.mode="$WANDB_MODE" ${METHODS:+"methods=$METHODS"} $EXTRA
