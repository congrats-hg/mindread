"""Training loop for Dialogue State Tracking models."""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.schema import INFORMABLE_SLOTS
from ..evaluation.metrics import DSTEvaluator, DSTMetrics

logger = logging.getLogger(__name__)


class DSTTrainer:
    """Trainer for Dialogue State Tracking models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        device: torch.device | None = None,
    ):
        """
        Initialize trainer.

        Args:
            model: The DST model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            config: Training configuration dictionary.
            device: Device to train on. Auto-detected if None.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 2e-5),
            weight_decay=config.get("weight_decay", 0.01),
        )

        # Scheduler
        num_training_steps = len(train_loader) * config.get("num_epochs", 10)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.get("learning_rate", 2e-5),
            total_steps=num_training_steps,
            pct_start=config.get("warmup_ratio", 0.1),
        )

        # Loss function (cross-entropy for each slot)
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision
        self.use_amp = config.get("mixed_precision", True) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler() if self.use_amp else None

        # Gradient accumulation
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

        # Tracking
        self.global_step = 0
        self.best_joint_accuracy = 0.0
        self.epochs_without_improvement = 0

        # W&B logging
        self.use_wandb = config.get("use_wandb", False)
        self.wandb_run = None

    def setup_wandb(self, project: str, name: str, config: dict[str, Any]) -> None:
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=project,
                name=name,
                config=config,
            )
            self.use_wandb = True
        except ImportError:
            logger.warning("wandb not installed. Logging disabled.")
            self.use_wandb = False

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to console and W&B."""
        if self.use_wandb and self.wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step or self.global_step)

    def compute_loss(
        self,
        logits: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute total loss across all slots."""
        total_loss = torch.tensor(0.0, device=self.device)

        for slot in INFORMABLE_SLOTS:
            slot_logits = logits[slot]
            slot_labels = labels[f"label_{slot}"]
            slot_loss = self.criterion(slot_logits, slot_labels)
            total_loss = total_loss + slot_loss

        return total_loss / len(INFORMABLE_SLOTS)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}",
            leave=False,
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            # Forward pass
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

                # Prepare labels
                labels = {
                    f"label_{slot}": batch[f"label_{slot}"].to(self.device)
                    for slot in INFORMABLE_SLOTS
                }

                loss = self.compute_loss(logits, labels)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get("max_grad_norm", 1.0),
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get("max_grad_norm", 1.0),
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log periodically
            if self.global_step % self.config.get("log_every_n_steps", 50) == 0:
                self.log_metrics(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                    }
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"train/epoch_loss": avg_loss}

    @torch.no_grad()
    def evaluate(self) -> DSTMetrics:
        """Evaluate on validation set."""
        self.model.eval()
        evaluator = DSTEvaluator()

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            # Forward pass
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            # Get predictions
            predictions = {slot: logits[slot].argmax(dim=-1) for slot in INFORMABLE_SLOTS}

            # Get labels
            labels = {
                slot: batch[f"label_{slot}"].to(self.device) for slot in INFORMABLE_SLOTS
            }

            # Update evaluator
            evaluator.update(
                predictions=predictions,
                labels=labels,
                dialogue_ids=batch.get("dialogue_ids"),
                turn_indices=batch.get("turn_indices"),
            )

        return evaluator.compute()

    def save_checkpoint(self, path: Path, epoch: int, metrics: DSTMetrics) -> None:
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_joint_accuracy": self.best_joint_accuracy,
            "metrics": metrics.to_dict(),
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> int:
        """Load model checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_joint_accuracy = checkpoint["best_joint_accuracy"]

        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint["epoch"]

    def train(
        self,
        num_epochs: int | None = None,
        checkpoint_dir: Path | None = None,
    ) -> DSTMetrics:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train. Uses config if None.
            checkpoint_dir: Directory to save checkpoints.

        Returns:
            Final validation metrics.
        """
        if num_epochs is None:
            num_epochs = self.config.get("num_epochs", 10)

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        early_stopping_patience = self.config.get("early_stopping_patience", 3)
        best_metrics = DSTMetrics()

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_metrics['train/epoch_loss']:.4f}")

            # Evaluate
            val_metrics = self.evaluate()
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Joint Goal Accuracy: {val_metrics.joint_goal_accuracy:.4f}"
            )

            # Log metrics
            all_metrics = {**train_metrics, **val_metrics.to_dict()}
            self.log_metrics(all_metrics)

            # Check for improvement
            if val_metrics.joint_goal_accuracy > self.best_joint_accuracy:
                self.best_joint_accuracy = val_metrics.joint_goal_accuracy
                self.epochs_without_improvement = 0
                best_metrics = val_metrics

                # Save best model
                if checkpoint_dir is not None:
                    self.save_checkpoint(
                        checkpoint_dir / "best_model.pt",
                        epoch,
                        val_metrics,
                    )
            else:
                self.epochs_without_improvement += 1

            # Save periodic checkpoint
            if checkpoint_dir is not None and (epoch + 1) % self.config.get("save_every_n_epochs", 1) == 0:
                self.save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                    epoch,
                    val_metrics,
                )

            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

        # Final summary
        logger.info(f"Training complete. Best Joint Goal Accuracy: {self.best_joint_accuracy:.4f}")

        if self.use_wandb and self.wandb_run is not None:
            import wandb

            wandb.finish()

        return best_metrics
