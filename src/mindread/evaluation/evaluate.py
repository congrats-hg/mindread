"""Evaluation script for trained DST models."""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..data.dataset import DSTC2Dataset, SLOT_VOCAB, collate_fn
from ..data.schema import INFORMABLE_SLOTS
from ..models.dst import create_model
from .metrics import DSTEvaluator

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: Path,
    model_type: str = "bert",
    model_name: str = "bert-base-uncased",
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load a trained model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = {"additional_special_tokens": ["[SYS]", "[USR]"]}
    tokenizer.add_special_tokens(special_tokens)

    # Create model
    model = create_model(model_type=model_type, model_name=model_name)
    model.encoder.resize_token_embeddings(len(tokenizer))

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    return model, tokenizer


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataset."""
    model.eval()
    evaluator = DSTEvaluator()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            predictions = {slot: logits[slot].argmax(dim=-1) for slot in INFORMABLE_SLOTS}
            labels = {slot: batch[f"label_{slot}"].to(device) for slot in INFORMABLE_SLOTS}

            evaluator.update(
                predictions=predictions,
                labels=labels,
                dialogue_ids=batch.get("dialogue_ids"),
                turn_indices=batch.get("turn_indices"),
            )

    metrics = evaluator.compute()
    errors = evaluator.get_error_analysis()

    return {
        "metrics": metrics.to_dict(),
        "errors": errors,
        "num_errors": len(errors),
    }


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained DST model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="bert",
        choices=["bert", "hierarchical"],
        help="Type of model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Pre-trained model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, tokenizer = load_model(
        Path(args.checkpoint),
        model_type=args.model_type,
        model_name=args.model_name,
        device=device,
    )

    # Load dataset
    dataset = DSTC2Dataset(
        tokenizer=tokenizer,
        split=args.split,
        data_dir=args.data_dir,
    )
    logger.info(f"Loaded {len(dataset)} samples from {args.split} split")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Evaluate
    results = evaluate_model(model, dataloader, device)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"\nJoint Goal Accuracy: {results['metrics']['joint_goal_accuracy']:.4f}")
    print("\nPer-slot Accuracy:")
    for slot in INFORMABLE_SLOTS:
        acc = results["metrics"][f"slot_accuracy/{slot}"]
        print(f"  {slot}: {acc:.4f}")
    print(f"\nTotal samples: {int(results['metrics']['num_samples'])}")
    print(f"Total errors: {results['num_errors']}")
    print("=" * 50)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert errors for JSON serialization
        results_json = {
            "metrics": results["metrics"],
            "num_errors": results["num_errors"],
            "split": args.split,
            "checkpoint": str(args.checkpoint),
        }

        with open(output_path, "w") as f:
            json.dump(results_json, f, indent=2)

        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
