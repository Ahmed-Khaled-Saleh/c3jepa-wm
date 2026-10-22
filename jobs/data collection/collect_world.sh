#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=collect_world
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=4:00:00
#SBATCH --output=./logs/out_%j_%x_%N.log  # includes job ID (j), job name (x) and node name (N)
#SBATCH --error=./logs/err_%j_%x_%N.err

# The FindGoal dataset on the cluster (mains/collect_world.py, configs/collect/config.yaml), every stage in one job:
#   collect (one part per `chunk` episodes, `workers` = the job's CPUs at once)  ->  merge (dataset.h5)  ->  trajectories
# The config's 260 layouts x 40 episodes, with stable-marl's MixedPolicy; the POV images draw the other agent
# (stable-marl 5a62dd0). Parts already written are skipped, so the same job can be submitted again if it runs out
# of time. Override anything at submission, e.g.
#   sbatch collect_world.sh                                          # every stage, into OUT_DIR
#   EXTRA="layouts=[0,4] episodes_per_layout=2" OUT_DIR=/scratch/project_2009050/datasets/findgoal_test sbatch --time=0:30:00 collect_world.sh
#   STAGE=merge sbatch collect_world.sh                              # one stage
# As an array instead (shard.index / shard.count come from the array, in the config), then the merge once:
#   STAGE=collect sbatch --array=0-3 collect_world.sh  and then  STAGE=merge sbatch collect_world.sh

STAGE=${STAGE:-all}
OUT_DIR=${OUT_DIR:-/scratch/project_2009050/datasets/findgoal_layouts}
EXTRA=${EXTRA:-}

export OMP_NUM_THREADS=1                  # one env per process: the processes are the parallelism
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module --force purge
module load python-pytorch/2.10
source /scratch/project_2009050/rl/bin/activate
cd /projappl/project_2009050/c3jepa-wm/mains/
mkdir -p logs "$OUT_DIR"

echo "------------------------------------------------"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME: stage=$STAGE out_dir=$OUT_DIR workers=$SLURM_CPUS_PER_TASK $EXTRA"
echo "------------------------------------------------"

srun python collect_world.py stage="$STAGE" out_dir="$OUT_DIR" $EXTRA
