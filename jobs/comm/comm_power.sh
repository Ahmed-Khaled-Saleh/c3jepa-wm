#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=comm_power
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=./logs/out_%j_%x_%N.log  # includes job ID (j), job name (x) and node name (N)
#SBATCH --error=./logs/err_%j_%x_%N.err

# The learned power control on Roihu, every stage in one job (mains/comm_power.py, configs/comm/config.yaml):
#   collect (CPU: trajectories)  ->  trace (GPU: Sionna RT, the visited pairs)  ->  train  ->  evaluate (+ Monte Carlo)
# Each stage skips what is done already (trajectories, traced pairs), so the same job can be submitted again if it
# runs out of time. Override anything at submission, e.g.
#   sbatch comm_power.sh                                         # every stage, the config's 260 layouts
#   STAGE=trace sbatch comm_power.sh                             # one stage
#   EXTRA="monte_carlo.runs=20 link.fading=rayleigh" STAGE=evaluate sbatch comm_power.sh
#   EXTRA="layouts.train=[0,1] layouts.calibration=[1,1] layouts.test=[1,1]" STAGE=trace sbatch --time=1:00:00 comm_power.sh
#                                                                # time one layout's tracing first
# Tracing as an array instead (then STAGE=train and STAGE=evaluate once, after all of it):
#   STAGE=collect sbatch --array=0-9 comm_power.sh  and  STAGE=trace sbatch --array=0-9 comm_power.sh
#   (shard.index / shard.count come from the array)

STAGE=${STAGE:-all}
DATA_DIR=${DATA_DIR:-/scratch/project_2009050/datasets/findgoal/comm_power}
WANDB_MODE=${WANDB_MODE:-online}          # offline if the compute nodes can't reach wandb.ai: then `wandb sync $DATA_DIR/wandb/offline-run-*`
EXTRA=${EXTRA:-}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}   # scikit-learn's gradient boosting (train, evaluate) uses them
export MPLBACKEND=Agg

module --force purge
module load python-pytorch/2.10
source /scratch/project_2009050/rl/bin/activate
cd /projappl/project_2009050/c3jepa-wm/mains/
mkdir -p logs "$DATA_DIR"

SHARD=""
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    SHARD="shard.index=$SLURM_ARRAY_TASK_ID shard.count=$SLURM_ARRAY_TASK_COUNT"
fi

echo "------------------------------------------------"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME: stage=$STAGE data_dir=$DATA_DIR $SHARD $EXTRA"
nvidia-smi -L
# fail now, not hours in: Sionna RT must see the GPU (comm_power.py also refuses to trace without it: tracing.device=cuda)
if [ "$STAGE" = "all" ] || [ "$STAGE" = "trace" ]; then
    python -c "import drjit as dr, mitsuba as mi; assert dr.has_backend(dr.JitBackend.CUDA), 'Dr.Jit sees no CUDA device'; mi.set_variant('cuda_ad_mono_polarized'); import sionna.rt; print('Sionna RT', sionna.rt.__version__, 'on', mi.variant())" || exit 1
fi
echo "------------------------------------------------"

srun python comm_power.py stage="$STAGE" data_dir="$DATA_DIR" wandb.mode="$WANDB_MODE" \
     wandb.name="${STAGE}-${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}" $SHARD $EXTRA
