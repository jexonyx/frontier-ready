#!/usr/bin/env python3
"""
Train a GPT model.

Simple launch:
    python scripts/train.py --run-name my_run

DDP launch for e.g. 8 GPUs:
    torchrun --standalone --nproc_per_node=8 scripts/train.py --run-name my_run

Resume from checkpoint:
    python scripts/train.py --run-name my_run --resume log/my_run/model_10000.pt
"""
import os
import argparse
import torch
from torch.distributed import init_process_group, destroy_process_group

from nanogpt import GPT, GPTConfig, DataLoaderLite, Trainer, get_cosine_lr_schedule


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Train a GPT model')
    parser.add_argument('--run-name', type=str, default='default',
                       help='name for this run (determines log subdirectory)')
    parser.add_argument('--resume', type=str, default=None,
                       help='checkpoint path to resume from')
    parser.add_argument('--data-root', type=str, default='edu_fineweb10B',
                       help='root directory containing data shards')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='micro batch size')
    parser.add_argument('--seq-length', type=int, default=1024,
                       help='sequence length')
    parser.add_argument('--total-batch-size', type=int, default=524288,
                       help='total batch size in tokens')
    parser.add_argument('--max-lr', type=float, default=6e-4,
                       help='maximum learning rate')
    parser.add_argument('--min-lr-ratio', type=float, default=0.1,
                       help='minimum learning rate as ratio of max')
    parser.add_argument('--warmup-steps', type=int, default=715,
                       help='number of warmup steps')
    parser.add_argument('--max-steps', type=int, default=19073,
                       help='maximum training steps')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                       help='weight decay')
    parser.add_argument('--eval-interval', type=int, default=250,
                       help='steps between evaluation')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                       help='steps between checkpoints')
    parser.add_argument('--vocab-size', type=int, default=50304,
                       help='vocabulary size (padded to nice number)')
    parser.add_argument('--n-layer', type=int, default=12,
                       help='number of transformer layers')
    parser.add_argument('--n-head', type=int, default=12,
                       help='number of attention heads')
    parser.add_argument('--n-embd', type=int, default=768,
                       help='embedding dimension')
    parser.add_argument('--use-compile', action='store_true',
                       help='use torch.compile (may interfere with eval/generation)')
    args = parser.parse_args()

    # Set up DDP (distributed data parallel)
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        if master_process:
            print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # Set random seed
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Calculate gradient accumulation steps
    B = args.batch_size
    T = args.seq_length
    assert args.total_batch_size % (B * T * ddp_world_size) == 0, \
        "total_batch_size must be divisible by B * T * ddp_world_size"
    grad_accum_steps = args.total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {args.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # Create data loaders
    train_loader = DataLoaderLite(
        B=B, T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="train",
        data_root=args.data_root,
        verbose=master_process
    )
    val_loader = DataLoaderLite(
        B=B, T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
        data_root=args.data_root,
        verbose=master_process
    )

    # Set precision
    torch.set_float32_matmul_precision('high')

    # Create model
    config = GPTConfig(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd
    )
    model = GPT(config)
    model.to(device)

    if master_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"total parameters: {total_params:,}")

    # Create optimizer
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.max_lr,
        device_type=device_type,
        verbose=master_process
    )

    # Learning rate schedule
    min_lr = args.max_lr * args.min_lr_ratio
    lr_schedule = get_cosine_lr_schedule(
        max_lr=args.max_lr,
        min_lr=min_lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps
    )

    # Create trainer
    ddp_config = {
        'ddp': ddp,
        'ddp_rank': ddp_rank,
        'ddp_local_rank': ddp_local_rank,
        'ddp_world_size': ddp_world_size,
        'master_process': master_process,
    }

    log_dir = os.path.join("log", args.run_name)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        device_type=device_type,
        ddp_config=ddp_config,
        log_dir=log_dir,
        max_steps=args.max_steps,
        grad_accum_steps=grad_accum_steps,
        learning_rate_schedule=lr_schedule,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        use_compile=args.use_compile,
    )

    # Train
    trainer.train(resume_from=args.resume)

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
