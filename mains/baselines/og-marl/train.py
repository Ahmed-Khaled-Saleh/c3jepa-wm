from omegaconf import DictConfig
import hydra

from buffer import OfflineReplayBuffer
from dataset import FindGoalOfflineDataset
from env_wrapper import OGMARLFindGoalWrapper
from module import ViTIQLCQLSystem

import jax
import torch
from og_marl.loggers import WandbLogger

from loguru import logger as loguru_logger





@hydra.main(version_base=None, config_path="configs", config_name="iql_cql")
def run_experiment(cfg: DictConfig) -> None:
    print(cfg)

    jax.config.update('jax_platform_name', 'cpu')

    dataset = FindGoalOfflineDataset(cfg.data_path)  # (a)
    buffer = OfflineReplayBuffer(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)           # (a), wrapped by (i)
    # buffer = OfflineReplayBuffer(dataset, batch_size=2, num_workers=0)

    wandb_config = {
        "system": cfg["system_name"],
        "seed": cfg["seed"],
        "training_steps": cfg["training_steps"],
        **cfg["task"],
        **cfg["replay"],
        **cfg["system"],
    }
    wandb_logger = WandbLogger(project=cfg["wandb_project"], config=wandb_config)

    env = OGMARLFindGoalWrapper(num_agents=2, max_steps=150, joint_reward=True)  # (c)
    system = ViTIQLCQLSystem(env, wandb_logger, hidden_dim=cfg.embed_dim)                        # (b)

    torch.manual_seed(cfg["seed"])
    loguru_logger.info("Starting training...")

    system.train(buffer, training_steps= int(cfg.training_steps))  # BaseOfflineSystem.train(), unmodified



if __name__ == "__main__":
    run_experiment()