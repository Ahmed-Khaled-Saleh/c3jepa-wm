import os
import random
from pathlib import Path
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from dotenv import load_dotenv
from loguru import logger
from c3jepa_wm.utils import init_data, init_model, init_trainer
# from c3jepa_wm.loggers.base import SlurmSafeLogger

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 1. Point hydra to your configuration folder and master file
@hydra.main(version_base=None, config_path="../configs/train/", config_name="config")
def main(cfg: DictConfig):
    # Optional: logger.info config block in terminal to confirm changes at runtime
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    load_dotenv("../.env")  # Load environment variables from .env file (e.g., API keys)
    # --- 2. Seed and Environment Setup ---
    seed_everything(cfg.exp_params.manual_seed)
    device = torch.device("cuda") #if torch.cuda.is_available() else "cpu" #: remove for traiing on puhti
    logger.info(f"Using runtime hardware device: {device}")

    # --- 3. Initialize Weights & Biases (Using Hydra Config Values) ---
    # Hydra creates an isolated working directory for every run automatically.
    # We resolve the save directory path to make sure logs land safely.
    slurm_jobid = os.getenv("SLURM_JOB_ID", "local_run")
    logger.info(f"SLURM_JOB_ID: {slurm_jobid}")

    wandb.init( 
        name="wm",
        project= cfg.logging_params.project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        # mode="offline",
        # settings=wandb.Settings(start_method="thread")
    )

    logger.info("Weights & Biases Initialized Successfully!")
    # --- 4. Setup Data Components ---
    # Pass parameters straight from Hydra's dataset group dictionary
    data_module = init_data(cfg)
    logger.info("Data Module Initialized Successfully!")

    # --- 5. Build Model Architecture ---
    # Filter out 'name' so it matches your architecture's constructor signature
    model = init_model(cfg)
    logger.info("Model Initialized Successfully!")

    # --- 6. Initialize Training Manager Class ---
    # (Reuses the single-GPU VQVAETrainer class created previously)
    trainer = init_trainer(cfg, data_module, model, device, slurm_jobid)
    logger.info("Trainer Initialized Successfully!")

    # --- 7. Execution Loop ---
    trainer.fit(cfg)
   

    wandb.finish()
    # logger.finish()

if __name__ == "__main__":
    main()