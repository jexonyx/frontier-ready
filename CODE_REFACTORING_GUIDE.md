# Code Reorganization Guide: Karpathy's Code → nanogpt Package

## Simplified Approach

We're taking Karpathy's code as a **starting point** and reorganizing it into our desired package structure. There's no need to preserve the original structure or maintain compatibility - **this is our code now** (with acknowledgment).

**Before:** Monolithic script (`train_gpt2.py`)
**After:** Modular package organized however we want

## Source Code

`train_gpt2.py` is a ~650 line monolithic script containing:
- Model classes (GPT, Block, MLP, CausalSelfAttention)
- Training loop
- Data loading
- Evaluation
- CLI arguments
- Helper functions

**Our approach:** Copy this code and reorganize it - no need to carefully extract or preserve structure!

## Key Principle

**Freedom to reorganize** - We're not trying to preserve Karpathy's structure or maintain upstream compatibility. Just organize the code in a way that makes sense for our experiments.

- ✅ Reorganize classes and functions as needed
- ✅ Rename things if it helps clarity
- ✅ Combine or split modules as makes sense
- ✅ Refactor into cleaner abstractions
- ❌ Don't worry about "preserving structure"
- ❌ Don't worry about "maintaining compatibility"

**It's our code now - make it work for us!**

---

## Target Package Structure

```
packages/nanogpt/
├── nanogpt/
│   ├── __init__.py          # Public API exports
│   ├── model.py             # Model classes
│   ├── config.py            # Configuration dataclasses
│   ├── training.py          # Trainer class
│   ├── data.py              # Data loading utilities
│   ├── eval.py              # Evaluation functions
│   ├── utils.py             # Helper functions
│   └── variants/
│       ├── __init__.py
│       └── rope.py          # RoPE variant
│
├── scripts/
│   ├── train.py             # CLI entry point
│   ├── prepare_data.py      # Data preparation
│   └── evaluate.py          # Evaluation CLI
│
└── pyproject.toml
```

---

## Reorganization Plan: Moving Code to Modules

### Model Classes → `model.py`

**Copy from train_gpt2.py:**
```python
# train_gpt2.py lines ~17-78
class CausalSelfAttention(nn.Module):
    ...

class MLP(nn.Module):
    ...

class Block(nn.Module):
    ...

class GPT(nn.Module):
    ...
```

**New file: `nanogpt/model.py`**
```python
"""
GPT-2 model implementation.

Contains the core transformer architecture: GPT, Block, MLP, and Attention.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect

from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with flash attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # ... rest of implementation ...


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # ... rest of implementation ...


class Block(nn.Module):
    """Transformer block: attention + MLP with residual connections."""

    # Class attributes for variant support
    attn_cls = CausalSelfAttention
    mlp_cls = MLP
    norm_cls = nn.LayerNorm

    def __init__(self, config: GPTConfig):
        super().__init__()
        # ... rest of implementation ...


class GPT(nn.Module):
    """GPT-2 language model."""

    # Class attributes for variant support
    block_cls = Block
    norm_cls = nn.LayerNorm

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([self.block_cls(config) for _ in range(config.n_layer)]),
            ln_f = self.norm_cls(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        # ... implementation ...

    def forward(self, idx, targets=None):
        """Forward pass."""
        # ... implementation ...

    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT-2 weights from HuggingFace."""
        # ... implementation ...
```

---

### Lines 79-86: Configuration → `config.py`

**Copy from train_gpt2.py:**
```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
```

**New file: `nanogpt/config.py`**
```python
"""
Configuration classes for GPT models and training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GPTConfig:
    """GPT model configuration."""
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_config: GPTConfig = field(default_factory=GPTConfig)

    # Data
    data_dir: Path = Path("data/edu_fineweb10B")
    train_split: str = "train"
    val_split: str = "val"

    # Training hyperparameters
    max_steps: int = 19072
    batch_size: int = 64
    sequence_length: int = 1024
    gradient_accumulation_steps: int = 16

    # Optimization
    learning_rate: float = 6e-4
    warmup_steps: int = 715
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Evaluation
    eval_interval: int = 250
    eval_iters: int = 20
    eval_hellaswag: bool = True

    # Checkpointing
    output_dir: Path = Path("outputs/default")
    checkpoint_interval: int = 5000

    # Hardware
    device: str = "cuda"
    compile: bool = True
    use_ddp: bool = False

    # Logging
    log_interval: int = 10

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
```

---

### Lines 220-400: Data Loading → `data.py`

**Copy from train_gpt2.py:**
```python
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root):
        ...
```

**New file: `nanogpt/data.py`**
```python
"""
Data loading utilities for GPT training.
"""

import os
import numpy as np
import torch
from pathlib import Path


class DataLoaderLite:
    """
    Lightweight data loader for GPT training.

    Loads tokenized data from sharded files and yields batches.
    """

    def __init__(
        self,
        batch_size: int,
        sequence_length: int,
        process_rank: int,
        num_processes: int,
        split: str,
        data_dir: Path,
    ):
        self.B = batch_size
        self.T = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Load shard files
        data_root = Path(data_dir)
        shards = sorted(list(data_root.glob(f"{split}_*.bin")))
        shards = [str(s) for s in shards]
        assert len(shards) > 0, f"No shards found for split {split}"

        self.shards = shards
        self.current_shard = 0
        self.tokens = self._load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def _load_tokens(self, filename: str) -> torch.Tensor:
        """Load tokens from shard file."""
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of data."""
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # Advance position
        self.current_position += B * T * self.num_processes

        # Load next shard if needed
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self._load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y
```

---

### Lines 400-600: Training Loop → `training.py`

**New file: `nanogpt/training.py`**
```python
"""
Training orchestration for GPT models.
"""

import os
import time
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import TrainingConfig
from .model import GPT
from .data import DataLoaderLite
from .eval import evaluate_loss, evaluate_hellaswag


class Trainer:
    """
    Trainer for GPT models.

    Handles:
    - Training loop
    - Optimization
    - Checkpointing
    - Evaluation
    - Logging
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Setup device
        self.setup_device()

        # Setup distributed training
        self.setup_distributed()

        # Create model
        self.model = self.create_model()

        # Setup data loaders
        self.train_loader = self.create_dataloader("train")
        self.val_loader = self.create_dataloader("val")

        # Setup optimizer
        self.optimizer = self.create_optimizer()

        # Setup output directory
        self.setup_output_dir()

    def setup_device(self):
        """Configure device (CPU/GPU)."""
        self.device_type = "cuda" if self.config.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)

    def setup_distributed(self):
        """Setup distributed training if enabled."""
        if self.config.use_ddp:
            # DDP setup
            assert torch.cuda.is_available()
            dist.init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.master_process = True

    def create_model(self) -> GPT:
        """Create and initialize model."""
        model = GPT(self.config.model_config)
        model.to(self.device)

        # Compile if enabled
        if self.config.compile:
            model = torch.compile(model)

        # Wrap in DDP if distributed
        if self.config.use_ddp:
            model = DDP(model, device_ids=[self.ddp_local_rank])

        return model

    def create_dataloader(self, split: str) -> DataLoaderLite:
        """Create data loader for given split."""
        return DataLoaderLite(
            batch_size=self.config.batch_size,
            sequence_length=self.config.sequence_length,
            process_rank=self.ddp_rank,
            num_processes=self.ddp_world_size,
            split=split,
            data_dir=self.config.data_dir,
        )

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Get parameters
        raw_model = self.model.module if self.config.use_ddp else self.model

        # Separate parameters for weight decay
        param_dict = {pn: p for pn, p in raw_model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Create optimizer
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        return optimizer

    def setup_output_dir(self):
        """Create output directory and log files."""
        if self.master_process:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.config.output_dir / "log.txt"
            self.metrics_file = self.config.output_dir / "metrics.jsonl"

    def get_lr(self, step: int) -> float:
        """Get learning rate for given step (with warmup and cosine decay)."""
        # Warmup
        if step < self.config.warmup_steps:
            return self.config.max_lr * (step + 1) / self.config.warmup_steps

        # Cosine decay
        if step > self.config.max_steps:
            return self.config.min_lr

        decay_ratio = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.max_lr - self.config.min_lr)

    def train_step(self, step: int):
        """Execute one training step."""
        # Set learning rate
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation
        self.optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(self.config.gradient_accumulation_steps):
            x, y = self.train_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                logits, loss = self.model(x, y)

            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps
            loss_accum += loss.detach()
            loss.backward()

        # Clip gradients
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        # Optimizer step
        self.optimizer.step()

        return loss_accum, norm, lr

    def train(self):
        """Main training loop."""
        for step in range(self.config.max_steps):
            # Training step
            t0 = time.time()
            loss, norm, lr = self.train_step(step)
            dt = time.time() - t0

            # Compute metrics
            tokens_per_sec = (
                self.config.batch_size
                * self.config.sequence_length
                * self.config.gradient_accumulation_steps
                * self.ddp_world_size
            ) / dt

            # Evaluation
            val_loss = None
            hella_acc = None
            if step % self.config.eval_interval == 0:
                val_loss = evaluate_loss(self.model, self.val_loader, self.config.eval_iters, self.device)
                if self.config.eval_hellaswag:
                    hella_acc = evaluate_hellaswag(self.model, self.device)

            # Logging
            if self.master_process:
                self.log_metrics(step, loss, val_loss, hella_acc, lr, norm, dt, tokens_per_sec)

            # Checkpointing
            if step % self.config.checkpoint_interval == 0 and self.master_process:
                self.save_checkpoint(step)

    def log_metrics(self, step, train_loss, val_loss, hella_acc, lr, norm, dt, tokens_per_sec):
        """Log metrics to files."""
        # Console log
        print(f"step {step:5d} | loss: {train_loss:.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

        # JSON log
        metrics = {
            "step": step,
            "train_loss": train_loss.item(),
            "lr": lr,
            "grad_norm": norm.item() if hasattr(norm, 'item') else norm,
            "dt_ms": dt * 1000,
            "tok_sec": tokens_per_sec,
        }
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        if hella_acc is not None:
            metrics["hella_acc"] = hella_acc

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def save_checkpoint(self, step):
        """Save model checkpoint."""
        checkpoint_path = self.config.output_dir / f"checkpoint_{step}.pt"
        raw_model = self.model.module if self.config.use_ddp else self.model

        torch.save({
            "step": step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, checkpoint_path)
```

---

### Lines 500-600: Evaluation → `eval.py`

**New file: `nanogpt/eval.py`**
```python
"""
Evaluation functions for GPT models.
"""

import torch
from pathlib import Path

# Import hellaswag utilities
from .hellaswag_utils import render_example, iterate_examples


def evaluate_loss(
    model: torch.nn.Module,
    dataloader,
    num_iters: int,
    device: torch.device,
) -> float:
    """
    Evaluate model loss on validation set.

    Args:
        model: The model to evaluate
        dataloader: Data loader for validation set
        num_iters: Number of batches to evaluate
        device: Device to run evaluation on

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(num_iters):
            x, y = dataloader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, loss = model(x, y)

            total_loss += loss.item()

    model.train()
    return total_loss / num_iters


def evaluate_hellaswag(
    model: torch.nn.Module,
    device: torch.device,
    data_dir: Path = Path("data/hellaswag"),
) -> float:
    """
    Evaluate model on HellaSwag benchmark.

    Args:
        model: The model to evaluate
        device: Device to run evaluation on
        data_dir: Directory containing hellaswag_val.jsonl

    Returns:
        Accuracy on HellaSwag
    """
    model.eval()
    num_correct = 0
    num_total = 0

    # Load HellaSwag data
    hellaswag_file = data_dir / "hellaswag_val.jsonl"

    with torch.no_grad():
        for example in iterate_examples(str(hellaswag_file)):
            # Render example and get tokens
            tokens, mask, label = render_example(example)
            tokens = torch.tensor(tokens, dtype=torch.long).to(device)

            # Get model predictions
            logits = model(tokens.unsqueeze(0))
            # ... (rest of evaluation logic)

            num_total += 1

    model.train()
    return num_correct / num_total if num_total > 0 else 0.0
```

---

### CLI Entry Point: `scripts/train.py`

**New file: `scripts/train.py`**
```python
#!/usr/bin/env python3
"""
CLI entry point for training GPT models.
"""

import argparse
from pathlib import Path

from nanogpt import Trainer
from nanogpt.config import TrainingConfig, GPTConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GPT model")

    # Model config
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)

    # Training config
    parser.add_argument("--data-dir", type=Path, default=Path("data/edu_fineweb10B"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/default"))
    parser.add_argument("--max-steps", type=int, default=19072)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=6e-4)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no-compile", action="store_false", dest="compile")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create config
    model_config = GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    training_config = TrainingConfig(
        model_config=model_config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        compile=args.compile,
    )

    # Create trainer
    trainer = Trainer(training_config)

    # Run training
    print("Starting training...")
    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()
```

---

## Import Update Strategy

### Before (monolithic):
```python
# Just run the script
python train_gpt2.py
```

### After (package):
```python
# As package
from nanogpt import GPT, GPTConfig, Trainer, TrainingConfig

# As CLI
nanogpt-train --data-dir data/edu_fineweb10B --output-dir outputs/baseline
```

---

## Testing Strategy

### Phase 1: Unit Tests
Test each module independently:

```python
# test_model.py
def test_model_forward():
    from nanogpt import GPT, GPTConfig

    config = GPTConfig()
    model = GPT(config)

    x = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(x, x)

    assert logits.shape == (2, 16, config.vocab_size)
```

### Phase 2: Integration Test
Test full training loop:

```python
# test_training.py
def test_trainer():
    from nanogpt import Trainer, TrainingConfig

    config = TrainingConfig(max_steps=10)
    trainer = Trainer(config)
    trainer.train()  # Should complete without errors
```

### Phase 3: Comparison Test
Verify outputs match original:

```bash
# Run original
python old/train_gpt2.py --max-steps 10 > old_output.txt

# Run new
nanogpt-train --max-steps 10 > new_output.txt

# Compare
diff old_output.txt new_output.txt
```

---

## Migration Execution Order

1. **Create `config.py`** (simple, no dependencies)
2. **Create `model.py`** (depends on config)
3. **Create `data.py`** (independent)
4. **Create `eval.py`** (depends on model)
5. **Create `training.py`** (depends on all above)
6. **Create `scripts/train.py`** (CLI wrapper)
7. **Test each module** as you create it
8. **Integration test** whole pipeline

---

## Common Pitfalls to Avoid

1. **Import cycles:** Be careful with circular imports
2. **Class attributes:** Preserve `block_cls`, `attn_cls` for variants
3. **Device handling:** Ensure device is passed correctly
4. **State dict compatibility:** Keep same keys for loading checkpoints
5. **DDP handling:** `model.module` vs `model` distinction

---

## Files Checklist

- [ ] `nanogpt/__init__.py`
- [ ] `nanogpt/config.py`
- [ ] `nanogpt/model.py`
- [ ] `nanogpt/data.py`
- [ ] `nanogpt/eval.py`
- [ ] `nanogpt/training.py`
- [ ] `nanogpt/utils.py`
- [ ] `nanogpt/variants/rope.py`
- [ ] `scripts/train.py`
- [ ] `scripts/prepare_data.py`
- [ ] `scripts/evaluate.py`
- [ ] `pyproject.toml`
- [ ] `README.md`

---

## Next Steps

1. Read through this guide
2. Start with `config.py` (easiest)
3. Work through files in order
4. Test after each file
5. Once all modules work, test integration
6. Compare with original to ensure correctness
