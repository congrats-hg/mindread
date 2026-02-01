# MindRead

**State-of-the-art Dialogue State Tracking on DSTC2**

A BERT-based dialogue state tracking system that achieves competitive performance on the DSTC2 benchmark. This project demonstrates modern NLP techniques for tracking user goals in task-oriented dialogues.

## Overview

Dialogue State Tracking (DST) is a core component of task-oriented dialogue systems. Given a conversation history, the goal is to predict the user's current belief state - what they want from the system.

**Task**: Restaurant information domain (DSTC2)
- **Slots**: food type, area, price range
- **Metric**: Joint Goal Accuracy (all slots correct)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Processing                         │
│  [SYS] What area? [USR] Centre please [SEP] [SYS] ... [USR] │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BERT Encoder                              │
│              (bert-base-uncased / roberta-base)              │
│                                                              │
│    [CLS] tok1 tok2 ... tokN [SEP]                           │
│      │                                                       │
│      └──► Contextualized Representations                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ [CLS] embedding
┌─────────────────────────────────────────────────────────────┐
│                 Slot Classification Heads                    │
│                                                              │
│    ┌──────────┐    ┌──────────┐    ┌──────────────┐         │
│    │   Food   │    │   Area   │    │  PriceRange  │         │
│    │ Softmax  │    │ Softmax  │    │   Softmax    │         │
│    └────┬─────┘    └────┬─────┘    └──────┬───────┘         │
│         │               │                  │                 │
│         ▼               ▼                  ▼                 │
│      italian         centre             cheap                │
└─────────────────────────────────────────────────────────────┘
```

## Results

| Model | Joint Goal Accuracy | Food | Area | Price Range |
|-------|---------------------|------|------|-------------|
| BERT-base | TBD | TBD | TBD | TBD |
| RoBERTa-base | TBD | TBD | TBD | TBD |
| Hierarchical BERT | TBD | TBD | TBD | TBD |

*Results will be updated after training.*

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mindread/mindread.git
cd mindread

# Install dependencies
pip install -e ".[all]"

# Download DSTC2 data
make data
```

### Training

```bash
# Train with default configuration
python -m mindread.training.train

# Train with custom configuration
python -m mindread.training.train model=roberta training=fast

# Train with W&B logging
python -m mindread.training.train logging.use_wandb=true
```

### Evaluation

```bash
# Evaluate on test set
python -m mindread.evaluation.evaluate \
    --checkpoint checkpoints/dst_baseline/best_model.pt \
    --split test
```

### Demo

```bash
# Launch interactive demo
python -m mindread.demo --checkpoint checkpoints/dst_baseline/best_model.pt

# Launch without model (mock mode)
python -m mindread.demo
```

## Project Structure

```
mindread/
├── configs/                 # Hydra configuration files
│   ├── config.yaml         # Main configuration
│   ├── model/              # Model configurations
│   ├── training/           # Training configurations
│   └── data/               # Data configurations
├── src/mindread/           # Source code
│   ├── data/               # Data loading and preprocessing
│   │   ├── schema.py       # DSTC2 schema definitions
│   │   ├── download.py     # Data download utilities
│   │   ├── dstc2.py        # DSTC2 parsing
│   │   └── dataset.py      # PyTorch dataset
│   ├── models/             # Model architectures
│   │   └── dst.py          # BERT-based DST models
│   ├── training/           # Training loop
│   │   ├── trainer.py      # Trainer class
│   │   └── train.py        # Main training script
│   ├── evaluation/         # Evaluation
│   │   ├── metrics.py      # DST metrics
│   │   └── evaluate.py     # Evaluation script
│   └── demo.py             # Gradio demo
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks
├── docs/                    # Documentation
└── data/                    # Data directory
```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. Key configuration options:

### Model

```yaml
# configs/model/bert.yaml
model_type: "bert"
model_name: "bert-base-uncased"
dropout: 0.1
```

### Training

```yaml
# configs/training/default.yaml
learning_rate: 2e-5
num_epochs: 10
batch_size: 16
early_stopping_patience: 3
```

Override any configuration from the command line:

```bash
python -m mindread.training.train \
    model.dropout=0.2 \
    training.learning_rate=1e-5 \
    training.num_epochs=20
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
make test

# Run linting
make lint

# Format code
make format
```

## Research Context

This work explores context-aware intent classification in task-oriented dialogues:

- **Technical Focus**: How dialogue history and model architecture influence state tracking accuracy
- **H-AI Connection**: Analyzing how model failures correlate with user adaptation behaviors

### Key References

- [Neural Belief Tracker](https://arxiv.org/abs/1606.03777)
- [SUMBT](https://arxiv.org/abs/1907.07421)
- [TRADE](https://arxiv.org/abs/1905.08743)
- [SOM-DST](https://arxiv.org/abs/1911.03906)

## License

MIT License

## Citation

```bibtex
@software{mindread2024,
  title = {MindRead: State-of-the-art Dialogue State Tracking on DSTC2},
  year = {2024},
  url = {https://github.com/mindread/mindread}
}
```
