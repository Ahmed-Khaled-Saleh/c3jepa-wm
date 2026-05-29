import os
import random
import yaml
import numpy as np
import torch
import wandb

from c3jepa_wm.models.msg_net import VQVAE
from c3jepa_wm.data.vae_datasets import VAEDataset
from c3jepa_wm.trainers.msg_trainer import VQVAETrainer

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # --- 1. Load Configuration File ---
    file_name = "/projappl/project_2009050/c3jepa-wm/nbs/exp/configs/vq_vae.yaml"
    with open(file_name, "r") as file:
        config = yaml.safe_load(file)

    config['data_params']['data_dir'] = "/scratch/project_2009050/datasets/dataset.h5"
    seed_everything(config["exp_params"]["manual_seed"])

    # --- 2. Determine Local HW Runtime Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using runtime hardware device: {device}")

    # --- 3. Initialize Tracking Environment (WandB) ---
    wandb.init(
        dir=config["logging_params"]["save_dir"],
        name=config["model_params"]["name"],
        project="vq-vae-gridworld",
        config=config,
    )
    os.makedirs("./checkpoints", exist_ok=True)

    # --- 4. Setup Data Components ---
    # Instantiates your Lightning Datamodule, extracts its native dataloaders
    data_module = VAEDataset(**config["data_params"], pin_memory=False)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # --- 5. Build Model Architecture ---
    model_params = {
        k: v for k, v in config["model_params"].items() if k != "name"
    }
    model = VQVAE(**model_params)

    # --- 6. Initialize Training Manager Class ---
    save_dir = config["logging_params"]["save_dir"]
    trainer = VQVAETrainer(model, config["exp_params"], device, save_dir)

    # --- 7. Execution Loop ---
    print(f"======= Training {config['model_params']['name']} =======")
    best_val_loss = float("inf")
    max_epochs = config["trainer_params"].get("max_epochs", 50)

    for epoch in range(1, max_epochs + 1):
        # Perform training and validation cycles
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate_epoch(val_loader, epoch)

        # Generate and save sample visual grid reconstructions
        trainer.sample_and_save_images(val_loader, epoch)

        # Build Checkpoint dictionary
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "val_loss": val_loss,
        }

        # Keep a rotating snapshot of the latest epoch
        torch.save(checkpoint_state, "./checkpoints/last.pt")

        # Top-performing metric evaluation checkpoint track (Replicates ModelCheckpoint)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"--> New best model found! Saving checkpoint...")
            torch.save(checkpoint_state, f"./checkpoints/best_vqvae.pt")

    # Clean up tracking logging process connection paths
    wandb.finish()


if __name__ == "__main__":
    main()