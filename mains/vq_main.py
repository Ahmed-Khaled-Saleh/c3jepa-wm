import os
import random
from pathlib import Path
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

from c3jepa_wm.data.data_module import VQDataModule
from c3jepa_wm.models.msg_net import VQVAE
from c3jepa_wm.trainers.msg_trainer import VQVAETrainer

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 1. Point hydra to your configuration folder and master file
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Optional: print config block in terminal to confirm changes at runtime
    print(OmegaConf.to_yaml(cfg))

    # --- 2. Seed and Environment Setup ---
    seed_everything(cfg.exp_params.manual_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using runtime hardware device: {device}")

    # --- 3. Initialize Weights & Biases (Using Hydra Config Values) ---
    # Hydra creates an isolated working directory for every run automatically.
    # We resolve the save directory path to make sure logs land safely.
    save_dir = Path(hydra.utils.to_absolute_path(cfg.logging_params.save_dir))
    save_dir.mkdir(exist_ok=True, parents=True)

    wandb.init(
        # dir=str(save_dir),
        name=cfg.model.name,
        project="vq-vae-gridworld",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    ckp_dir = cfg.logging_params.ckp_dir
    os.makedirs(ckp_dir, exist_ok=True)

    # --- 4. Setup Data Components ---
    # Pass parameters straight from Hydra's dataset group dictionary
    # data_params = OmegaConf.to_container(cfg.dataset, resolve=True)
    data_module = VQDataModule(cfg)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # --- 5. Build Model Architecture ---
    # Filter out 'name' so it matches your architecture's constructor signature
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_params = {k: v for k, v in model_config.items() if k != "name"}
    model = VQVAE(**model_params)

    # --- 6. Initialize Training Manager Class ---
    # (Reuses the single-GPU VQVAETrainer class created previously)
    trainer = VQVAETrainer(
        model=model,
        params=cfg.exp_params,
        device=device,
        save_dir=str(save_dir),
    )

    # --- 7. Execution Loop ---
    print(f"======= Training {cfg.model.name} =======")
    best_val_loss = float("inf")

    for epoch in range(1, cfg.exp_params.max_epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate_epoch(val_loader, epoch)

        trainer.sample_and_save_images(val_loader, epoch)

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint_state, f"{ckp_dir}/last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("--> New best model found! Saving checkpoint...")
            torch.save(checkpoint_state, f"{ckp_dir}/best_vqvae.pt")

    wandb.finish()


if __name__ == "__main__":
    main()