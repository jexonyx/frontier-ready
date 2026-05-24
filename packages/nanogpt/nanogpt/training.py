"""Training utilities and Trainer class for GPT models."""
import os
import math
import time
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import DataLoaderLite
from .eval import get_most_likely_row, iterate_examples, render_example


class Trainer:
    """Handles GPT model training with DDP support."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        device_type,
        ddp_config=None,
        log_dir="log",
        max_steps=19073,
        grad_accum_steps=1,
        learning_rate_schedule=None,
        eval_interval=250,
        checkpoint_interval=1000,
        use_compile=False,
    ):
        """
        Initialize the Trainer.

        Args:
            model: The GPT model to train
            train_loader: DataLoaderLite for training data
            val_loader: DataLoaderLite for validation data
            optimizer: Optimizer (e.g., AdamW)
            device: Device to train on
            device_type: 'cuda' or 'cpu'
            ddp_config: Dict with DDP settings (ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process)
            log_dir: Directory to save logs and checkpoints
            max_steps: Maximum training steps
            grad_accum_steps: Gradient accumulation steps
            learning_rate_schedule: Function that takes step and returns learning rate
            eval_interval: Steps between evaluation
            checkpoint_interval: Steps between checkpoints
            use_compile: Whether to use torch.compile
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.device_type = device_type
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.learning_rate_schedule = learning_rate_schedule
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.use_compile = use_compile

        # DDP configuration
        if ddp_config is None:
            ddp_config = {
                'ddp': False,
                'ddp_rank': 0,
                'ddp_local_rank': 0,
                'ddp_world_size': 1,
                'master_process': True,
            }
        self.ddp = ddp_config['ddp']
        self.ddp_rank = ddp_config['ddp_rank']
        self.ddp_local_rank = ddp_config['ddp_local_rank']
        self.ddp_world_size = ddp_config['ddp_world_size']
        self.master_process = ddp_config['master_process']

        # Wrap model with DDP if needed
        if self.ddp:
            self.model = DDP(model, device_ids=[self.ddp_local_rank])
        self.raw_model = self.model.module if self.ddp else self.model

        # Compile if requested
        if use_compile:
            self.model = torch.compile(self.model)

        # Logging
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "log.txt")
        self.metrics_file = os.path.join(log_dir, "metrics.jsonl")

        # Activation norm hooks
        self.log_activations = False
        self.activation_norms = {}
        self._setup_activation_hooks()

    def _setup_activation_hooks(self):
        """Setup hooks to monitor activation norms."""
        def make_activation_hook(layer_idx):
            def hook(module, input, output):
                if self.log_activations:
                    self.activation_norms[layer_idx] = output.detach().norm().item()
            return hook

        for layer_idx, block in enumerate(self.raw_model.transformer.h):
            block.register_forward_hook(make_activation_hook(layer_idx))

    def train(self, start_step=0, resume_from=None):
        """
        Run the training loop.

        Args:
            start_step: Step to start from (for resumption)
            resume_from: Checkpoint path to resume from
        """
        # Handle resumption
        if resume_from is not None:
            start_step = self._load_checkpoint(resume_from)

        # Initialize log files
        if start_step == 0:
            with open(self.log_file, "w") as f:
                pass
            with open(self.metrics_file, "w") as f:
                pass

        # Training loop
        for step in range(start_step, self.max_steps):
            t0 = time.time()
            last_step = (step == self.max_steps - 1)
            self.log_activations = (step % 50 == 0)
            self.activation_norms.clear()
            step_val_loss = None
            step_hella_acc = None

            # Validation
            if step % self.eval_interval == 0 or last_step:
                step_val_loss = self._validate()

            # HellaSwag evaluation
            if (step % self.eval_interval == 0 or last_step) and (not self.use_compile):
                step_hella_acc = self._evaluate_hellaswag()

            # Text generation
            if ((step > 0 and step % self.eval_interval == 0) or last_step) and (not self.use_compile):
                self._generate_samples(step)

            # Training step
            loss_accum, norm, lr, update_ratios = self._train_step(step)

            # Timing
            if self.device_type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0
            tokens_processed = self.train_loader.B * self.train_loader.T * self.grad_accum_steps * self.ddp_world_size
            tokens_per_sec = tokens_processed / dt

            # Logging
            if self.master_process:
                self._log_metrics(step, loss_accum, lr, norm, dt, tokens_per_sec,
                                step_val_loss, step_hella_acc, update_ratios)

                # Checkpointing
                if step > 0 and (step % self.checkpoint_interval == 0 or last_step):
                    self._save_checkpoint(step, step_val_loss)

    def _validate(self):
        """Run validation and return validation loss."""
        if self.device_type == "cuda":
            torch.cuda.empty_cache()

        self.model.eval()
        self.val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = self.val_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        if self.ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        val_loss = val_loss_accum.item()
        if self.master_process:
            print(f"validation loss: {val_loss:.4f}")
            with open(self.log_file, "a") as f:
                f.write(f"{self.current_step} val {val_loss:.4f}\n")

        return val_loss

    def _evaluate_hellaswag(self):
        """Run HellaSwag evaluation and return accuracy."""
        if self.device_type == "cuda":
            torch.cuda.empty_cache()

        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            if i % self.ddp_world_size != self.ddp_rank:
                continue

            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(self.device)
            mask = mask.to(self.device)

            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)

            num_total += 1
            num_correct_norm += int(pred_norm == label)

        # Reduce stats across processes
        if self.ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=self.device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=self.device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()

        acc_norm = num_correct_norm / num_total
        if self.master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(self.log_file, "a") as f:
                f.write(f"{self.current_step} hella {acc_norm:.4f}\n")

        return acc_norm

    def _generate_samples(self, step):
        """Generate and print sample text."""
        import tiktoken
        from torch.nn import functional as F

        if self.device_type == "cuda":
            torch.cuda.empty_cache()

        self.model.eval()
        enc = tiktoken.get_encoding("gpt2")
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(self.device)
        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(42 + self.ddp_rank)

        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)

        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {self.ddp_rank} sample {i}: {decoded}")

    def _train_step(self, step):
        """Execute one training step and return metrics."""
        self.current_step = step
        self.model.train()
        self.optimizer.zero_grad()
        loss_accum = 0.0

        # Snapshot parameters for update-to-weight ratios
        param_snapshots = None
        if step % 50 == 0:
            if self.device_type == "cuda":
                torch.cuda.empty_cache()
            param_snapshots = {}
            for layer_idx, block in enumerate(self.raw_model.transformer.h):
                params = torch.cat([p.data.detach().reshape(-1) for p in block.parameters()])
                param_snapshots[layer_idx] = params.clone()

        # Gradient accumulation
        for micro_step in range(self.grad_accum_steps):
            x, y = self.train_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)
            if self.ddp:
                self.model.require_backward_grad_sync = (micro_step == self.grad_accum_steps - 1)
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                logits, loss = self.model(x, y)
            loss = loss / self.grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if self.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Learning rate schedule
        if self.learning_rate_schedule is not None:
            lr = self.learning_rate_schedule(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self.optimizer.param_groups[0]['lr']

        self.optimizer.step()

        # Compute update-to-weight ratios
        update_ratios = None
        if param_snapshots is not None:
            update_ratios = {}
            for layer_idx, block in enumerate(self.raw_model.transformer.h):
                new_params = torch.cat([p.data.detach().reshape(-1) for p in block.parameters()])
                old_params = param_snapshots[layer_idx]
                update_ratios[layer_idx] = (new_params - old_params).norm().item() / old_params.norm().item()
            del param_snapshots

        return loss_accum, norm, lr, update_ratios

    def _log_metrics(self, step, loss_accum, lr, norm, dt, tokens_per_sec,
                     step_val_loss, step_hella_acc, update_ratios):
        """Log training metrics."""
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

        # GPU memory
        gpu_mem_gb = torch.cuda.memory_allocated() / 1e9 if self.device_type == "cuda" else 0.0

        # Per-layer gradient norms and weight norms
        layer_grad_norms = []
        layer_weight_norms = []
        for block in self.raw_model.transformer.h:
            grad_sq = sum(p.grad.norm().item()**2 for p in block.parameters() if p.grad is not None)
            weight_sq = sum(p.data.norm().item()**2 for p in block.parameters())
            layer_grad_norms.append(grad_sq**0.5)
            layer_weight_norms.append(weight_sq**0.5)

        # log.txt
        with open(self.log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f} lr {lr:.4e} norm {norm:.4f} dt {dt*1000:.2f} tok/sec {tokens_per_sec:.2f} gpu_mem {gpu_mem_gb:.2f}\n")

        # metrics.jsonl
        metrics = {
            "step": step,
            "train_loss": loss_accum.item(),
            "lr": lr,
            "grad_norm": norm.item() if hasattr(norm, 'item') else norm,
            "dt_ms": dt * 1000,
            "tok_sec": tokens_per_sec,
            "gpu_mem_gb": gpu_mem_gb,
            "layer_grad_norms": layer_grad_norms,
            "layer_weight_norms": layer_weight_norms,
        }
        if step_val_loss is not None:
            metrics["val_loss"] = step_val_loss
        if step_hella_acc is not None:
            metrics["hella_acc"] = step_hella_acc
        if update_ratios is not None:
            metrics["layer_update_ratios"] = [update_ratios[i] for i in range(len(self.raw_model.transformer.h))]
        if self.activation_norms:
            metrics["layer_act_norms"] = [self.activation_norms.get(i, 0.0) for i in range(len(self.raw_model.transformer.h))]

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def _save_checkpoint(self, step, val_loss):
        """Save a checkpoint."""
        if self.device_type == "cuda":
            torch.cuda.empty_cache()

        checkpoint_path = os.path.join(self.log_dir, f"model_{step:05d}.pt")
        checkpoint = {
            'model': {k: v.cpu() for k, v in self.raw_model.state_dict().items()},
            'config': vars(self.raw_model.config),
            'step': step,
            'val_loss': val_loss,
            'optimizer': {
                k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                for k, v in self.optimizer.state_dict().items()
            },
            'rng_state': torch.random.get_rng_state(),
        }
        if self.device_type == "cuda":
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path):
        """Load a checkpoint and return the step to resume from."""
        if self.master_process:
            print(f"resuming from checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        torch.random.set_rng_state(checkpoint['rng_state'])
        if self.device_type == "cuda" and 'cuda_rng_state' in checkpoint:
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

        start_step = checkpoint['step'] + 1
        if self.master_process:
            print(f"resumed at step {start_step}")

        del checkpoint
        return start_step


def get_cosine_lr_schedule(max_lr, min_lr, warmup_steps, max_steps):
    """
    Create a cosine learning rate schedule with warmup.

    Args:
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps

    Returns:
        Function that takes step and returns learning rate
    """
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)
    return get_lr
