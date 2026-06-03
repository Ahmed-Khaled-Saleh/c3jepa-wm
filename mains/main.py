import os
import random
from pathlib import Path
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from dotenv import load_dotenv

from c3jepa_wm.utils import init_data, init_model, init_trainer


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 1. Point hydra to your configuration folder and master file
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Optional: print config block in terminal to confirm changes at runtime
    print(OmegaConf.to_yaml(cfg, resolve=True))
    load_dotenv("../.env")  # Load environment variables from .env file (e.g., API keys)
    # --- 2. Seed and Environment Setup ---
    seed_everything(cfg.exp_params.manual_seed)
    device = torch.device("cuda") #if torch.cuda.is_available() else "cpu" : remove for traiing on puhti
    print(f"Using runtime hardware device: {device}")

    # --- 3. Initialize Weights & Biases (Using Hydra Config Values) ---
    # Hydra creates an isolated working directory for every run automatically.
    # We resolve the save directory path to make sure logs land safely.
    slurm_jobid = os.getenv("SLURM_JOB_ID", "local_run")
    print(f"SLURM_JOB_ID: {slurm_jobid}")

    wandb.init( 
        name="wm",
        project= cfg.logging_params.project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # --- 4. Setup Data Components ---
    # Pass parameters straight from Hydra's dataset group dictionary
    data_module = init_data(cfg)

    # --- 5. Build Model Architecture ---
    # Filter out 'name' so it matches your architecture's constructor signature
    model = init_model(cfg)

    # --- 6. Initialize Training Manager Class ---
    # (Reuses the single-GPU VQVAETrainer class created previously)
    trainer = init_trainer(cfg, data_module, model, device)

    # --- 7. Execution Loop ---
    for epoch in range(1, cfg.pipeline.max_epochs + 1):
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.validate_epoch(epoch)
        trainer.scheduler.step(val_loss)
        trainer.checkpoint(epoch, val_loss)

    wandb.finish()


if __name__ == "__main__":
    main()