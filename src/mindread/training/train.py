"""Main training script with Hydra configuration."""

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..data.dataset import DSTC2Dataset, collate_fn
from ..models.dst import create_model
from .trainer import DSTTrainer

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Main training function.

    Args:
        cfg: Hydra configuration.

    Returns:
        Best joint goal accuracy for hyperparameter optimization.
    """
    # Print config
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    set_seed(cfg.experiment.seed)

    # Determine device
    if cfg.training.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.training.device)
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    # Add special tokens for dialogue
    special_tokens = {"additional_special_tokens": ["[SYS]", "[USR]"]}
    tokenizer.add_special_tokens(special_tokens)

    # Load datasets
    logger.info("Loading datasets...")
    data_dir = Path(cfg.paths.data_dir)

    train_dataset = DSTC2Dataset(
        tokenizer=tokenizer,
        split="train",
        max_length=cfg.data.max_length,
        max_context_turns=cfg.data.max_context_turns,
        data_dir=str(data_dir),
    )

    val_dataset = DSTC2Dataset(
        tokenizer=tokenizer,
        split="dev",
        max_length=cfg.data.max_length,
        max_context_turns=cfg.data.max_context_turns,
        data_dir=str(data_dir),
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
    )

    # Create model
    logger.info(f"Creating model: {cfg.model.model_type}")
    model = create_model(
        model_type=cfg.model.model_type,
        model_name=cfg.model.model_name,
        dropout=cfg.model.dropout,
        **OmegaConf.to_container(cfg.model, resolve=True),
    )

    # Resize token embeddings for special tokens
    model.encoder.resize_token_embeddings(len(tokenizer))

    # Create trainer
    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    training_config["use_wandb"] = cfg.logging.use_wandb
    training_config["log_every_n_steps"] = cfg.logging.log_every_n_steps

    trainer = DSTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
    )

    # Setup W&B if enabled
    if cfg.logging.use_wandb:
        trainer.setup_wandb(
            project=cfg.logging.wandb_project,
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Train
    checkpoint_dir = Path(cfg.paths.checkpoint_dir) / cfg.experiment.name
    best_metrics = trainer.train(
        num_epochs=cfg.training.num_epochs,
        checkpoint_dir=checkpoint_dir,
    )

    logger.info(f"Training complete!")
    logger.info(f"Best Joint Goal Accuracy: {best_metrics.joint_goal_accuracy:.4f}")
    for slot, acc in best_metrics.slot_accuracy.items():
        logger.info(f"  {slot}: {acc:.4f}")

    return best_metrics.joint_goal_accuracy


if __name__ == "__main__":
    main()
