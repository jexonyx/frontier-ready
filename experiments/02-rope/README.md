# Experiment 02: RoPE Variant GPT-2 124M

GPT-2 124M with Rotary Position Embeddings (RoPE) instead of learned positional embeddings.

## Configuration

- **Model**: GPT-2 124M (12 layers, 12 heads, 768 embedding dim)
- **Data**: FineWeb-Edu 10B tokens
- **Training**: 19,072 steps (~10B tokens)
- **Batch size**: 524,288 tokens/step
- **Hardware**: Single A100 GPU

## Running

```bash
cd experiments/02-rope
uv run python run.py
```

## Outputs

- **Checkpoints**: `../../outputs/rope/checkpoints/`
- **Metrics**: `../../outputs/rope/metrics.jsonl`
- **Analysis**: `writeup/`

## Results

See `writeup/` directory for full analysis and figures.
