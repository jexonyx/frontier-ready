# Experiment 01: Baseline GPT-2 124M

Standard GPT-2 124M training from scratch.

## Configuration

- **Model**: GPT-2 124M (12 layers, 12 heads, 768 embedding dim)
- **Data**: FineWeb-Edu 10B tokens
- **Training**: 19,072 steps (~10B tokens)
- **Batch size**: 524,288 tokens/step
- **Hardware**: Single A100 GPU

## Running

```bash
cd experiments/01-baseline
uv run python run.py
```

## Outputs

- **Checkpoints**: `../../outputs/baseline/checkpoints/`
- **Metrics**: `../../outputs/baseline/metrics.jsonl`
- **Analysis**: `writeup/`

## Results

See `writeup/` directory for full analysis and figures.
