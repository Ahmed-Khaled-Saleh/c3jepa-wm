#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=vq_tokens
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=./logs/out_%j_%x_%N.log  # includes job ID (j), job name (x) and node name (N)
#SBATCH --error=./logs/err_%j_%x_%N.err

# What the VQ-VAE's tokens stand for (mains/vq_tokens.py, configs/analysis/vq_tokens.yaml): on the validation layouts'
# frames, the codes of each kind of tile, galleries, each code decoded alone, token swaps, token maps and the codebook.
# Saves <OUT_DIR>/tokens.npz and summary.json (the figures redraw from them: stage=plot), draws <OUT_DIR>/figures and
# logs all of it to wandb (the saved results as an artifact). Override anything at submission, e.g.
#   sbatch vq_tokens.sh
#   VQVAE=.../last_state.pt OUT_DIR=/scratch/project_2009050/expirements/findgoal/vq_tokens_last sbatch vq_tokens.sh
#   EXTRA="frames=50000 swap_cell=[4,3]" sbatch vq_tokens.sh
#   EXTRA="stage=plot" sbatch --time=0:20:00 vq_tokens.sh            # redraw only
# With more agents than it was trained on (the partner's colour no longer follows from the ego's): a test set of the
# validation layouts with 4 agents (stable-marl gives each its own colour), then the same analysis on it; the context
# figure and alone/<label> say whether each token still draws its tile by itself:
#   cd "../data collection" && EXTRA="env.agents=4 layouts=[200,260] episodes_per_layout=4" \
#       OUT_DIR=/scratch/project_2009050/datasets/findgoal_4agents sbatch --time=1:00:00 collect_world.sh
#   EXTRA="data_dir=/scratch/project_2009050/datasets/findgoal_4agents" \
#       OUT_DIR=/scratch/project_2009050/expirements/findgoal/vq_tokens_4agents sbatch vq_tokens.sh

VQVAE=${VQVAE:-/scratch/project_2009050/expirements/findgoal/checkpoints/phase-1-vqvae_pretraining/job_2462489/agent_vqvae_trainer/best_state_step_8_acc_1.0000.pt}
OUT_DIR=${OUT_DIR:-/scratch/project_2009050/expirements/findgoal/vq_tokens}
WANDB_MODE=${WANDB_MODE:-online}
EXTRA=${EXTRA:-}

export OMP_NUM_THREADS=8
export MPLBACKEND=Agg

module --force purge
module load python-pytorch/2.10
source /scratch/project_2009050/rl/bin/activate
cd /projappl/project_2009050/c3jepa-wm/mains/
mkdir -p logs

echo "------------------------------------------------"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME: vqvae=$VQVAE out_dir=$OUT_DIR $EXTRA"
echo "------------------------------------------------"

srun python vq_tokens.py checkpoint="$VQVAE" out_dir="$OUT_DIR" wandb.mode="$WANDB_MODE" $EXTRA
