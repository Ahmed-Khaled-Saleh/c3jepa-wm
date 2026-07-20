#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=WM_horizon_eval
#SBATCH --output=logs/horizon_%A_%a.out
#SBATCH --error=logs/horizon_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-14             # Number of algorithms (0 to N-1)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=36:00:00             # Adjust based on expected runtime


horizon=(10 15 20 25 30 35 40 45 50 55 60 70 80 90 100) 


# 3. Safety Check: Ensure the SLURM_ARRAY_TASK_ID is within bounds
if [ $SLURM_ARRAY_TASK_ID -ge ${#horizon[@]} ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID out of bounds. Max index is $((${#horizon[@]} - 1))"
    exit 1
fi

# 4. Extract parameters for this specific task
CURRENT_COMBO=${horizon[$SLURM_ARRAY_TASK_ID]}
read -r CURRENT_HORIZON <<< "$CURRENT_COMBO"

echo "------------------------------------------------"
echo "Job ID: $SLURM_ARRAY_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running: $CURRENT_HORIZON"
echo "------------------------------------------------"


# 3. Load your environment (Conda, modules, etc.)
# module load cuda
# source activate your_env
module --force purge
module load pytorch
source /scratch/project_2009050/rl/bin/activate
cd /projappl/project_2009050/c3jepa-wm/mains/


# 4. Launch Hydra
python eval.py pipeline.horizon=${CURRENT_HORIZON}