import os
import random
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from dotenv import load_dotenv
import gymnasium as gym
import multigrid.envs

env = gym.make('MultiGrid-FindGoal-15x15-v0', agents=2, render_mode='rgb_array', num_obstacles=6, width=15, height=15)




from c3jepa_wm.utils import init_data, init_model, init_evaluator

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 1. Point hydra to your configuration folder and master file
@hydra.main(version_base=None, config_path="../configs/eval/", config_name="config")
def main(cfg: DictConfig):
    # Optional: print config block in terminal to confirm changes at runtime
    print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    load_dotenv("../.env")  # Load environment variables from .env file (e.g., API keys)
    # --- 2. Seed and Environment Setup ---
    seed_everything(cfg.exp_params.manual_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu" #: remove for traiing on puhti
    print(f"Using runtime hardware device: {device}", flush=True)

    # --- 3. Initialize Weights & Biases (Using Hydra Config Values) ---
    # Hydra creates an isolated working directory for every run automatically.
    # We resolve the save directory path to make sure logs land safely.
    slurm_jobid = os.getenv("SLURM_JOB_ID", "local_run")
    print(f"SLURM_JOB_ID: {slurm_jobid}", flush=True)

    wandb.init( 
        name="wm",
        project= cfg.logging_params.project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    print("Weights & Biases Initialized Successfully!", flush=True)
    # --- 4. Setup Data Components ---
    # Pass parameters straight from Hydra's dataset group dictionary
    data_module = init_data(cfg)
    print("Data Module Initialized Successfully!", flush=True)

    # --- 5. Build Model Architecture ---
    model = init_model(cfg)
    print("Model Initialized Successfully!", flush=True)

    # --- 6. Initialize Training Manager Class ---
    # (Reuses the single-GPU VQVAETrainer class created previously)
    evaluator = init_evaluator(cfg, data_module, model, device, slurm_jobid)
    print("Evaluator Initialized Successfully!", flush=True)

    # --- 7. Execution Loop ---
    # evaluator.evaluate_dataset()
    hydra.utils.call(config= cfg.pipeline.eval_func, self= evaluator, env= env)
    
    wandb.finish()

if __name__ == "__main__":
    main()