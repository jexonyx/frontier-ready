# Directory Restructure Proposal

## Recommended Structure: Peer Experiments with Shared Base

```
frontier-ready/
├── build-nanogpt/           # Your fork (base training code)
│   ├── train_gpt2.py
│   ├── variant_rope.py
│   ├── fineweb.py
│   ├── hellaswag.py
│   ├── edu_fineweb10B/      # Data directory
│   └── pyproject.toml
│
├── experiments/
│   ├── baseline/
│   │   ├── logs/            # Training logs & checkpoints
│   │   ├── writeup/
│   │   ├── config.yaml      # Experiment configuration
│   │   └── README.md
│   │
│   └── rope/
│       ├── logs/
│       ├── writeup/
│       ├── config.yaml
│       └── README.md
│
├── shared/
│   ├── analysis/            # parse_metrics.py, visualize_run.py, etc.
│   ├── pyproject.toml       # Shared dependencies (pandas, matplotlib, etc.)
│   └── README.md
│
├── infra/                   # Infrastructure code (unchanged)
└── PLAN.md
```

## Why This Structure?

### **Separation of Concerns**
- **`build-nanogpt/`** - Pure training code, can be updated/forked independently
- **`experiments/*/`** - Each experiment's artifacts (logs, writeup, config)
- **`shared/`** - Reusable analysis infrastructure

### **Scalability**
- Adding a new experiment = new folder in `experiments/`
- No code duplication
- Each experiment is self-documenting

### **Workflow Benefits**
```bash
# Run baseline
cd build-nanogpt
python train_gpt2.py --output ../experiments/baseline/logs

# Run RoPE variant
python train_gpt2.py --variant rope --output ../experiments/rope/logs

# Analyze any experiment
cd ../shared
python analysis/visualize_run.py \
    --log-dir ../experiments/baseline/logs \
    --output-dir ../experiments/baseline/writeup/figures
```

## Migration Steps

### 1. Move build-nanogpt to root
```bash
cd /Users/jex/frontier-ready
git mv study1-gpt/build-nanogpt ./build-nanogpt
# Update .gitmodules: path = build-nanogpt
```

### 2. Create experiment directories
```bash
mkdir -p experiments/baseline experiments/rope
```

### 3. Move experiment-specific content
```bash
# Move writeup
git mv study1-gpt/writeup experiments/baseline/

# Create logs directory
mkdir experiments/baseline/logs
# (Future: move or symlink log/ directories here)
```

### 4. Move shared infrastructure
```bash
mkdir -p shared
git mv study1-gpt/analysis shared/
git mv study1-gpt/pyproject.toml shared/
```

### 5. Create experiment README files
```bash
cat > experiments/baseline/README.md << 'EOF'
# Baseline GPT-2 124M

Standard GPT-2 124M training on FineWeb-Edu 10B tokens.

## Run Training
\`\`\`bash
cd ../../build-nanogpt
python train_gpt2.py --output ../experiments/baseline/logs
\`\`\`

## Analysis
See `writeup/` for results and analysis.
EOF
```

### 6. Update .gitignore
```bash
# Add to .gitignore
experiments/*/logs/*
!experiments/*/logs/.gitkeep
```

### 7. Clean up study1-gpt
```bash
# Remove now-empty study1-gpt directory
rm -rf study1-gpt
```

### 8. Update documentation
- Update `frontier-ready/README.md` to reflect new structure
- Update `build-nanogpt/README.md` to reference experiments directory

## File Moves Summary

| Current | New |
|---------|-----|
| `study1-gpt/build-nanogpt/` | `build-nanogpt/` |
| `study1-gpt/analysis/` | `shared/analysis/` |
| `study1-gpt/writeup/` | `experiments/baseline/writeup/` |
| `study1-gpt/pyproject.toml` | `shared/pyproject.toml` |

## Alternative: Keep It Simple (Option 4)

If full restructure feels like too much, minimal change:

```
frontier-ready/
├── study1-gpt/
│   ├── base/                # Renamed from build-nanogpt
│   ├── experiments/
│   │   ├── baseline/
│   │   └── rope/
│   ├── analysis/            # Shared
│   └── writeup/            # Shared docs
```

This keeps everything in `study1-gpt/` but makes experiments explicit.

## Decision Points

Before migrating, decide:

1. **Where should training outputs go?**
   - Inside `build-nanogpt/log/` (current)
   - Inside `experiments/*/logs/` (recommended)
   - Separate `outputs/` directory

2. **Where should data live?**
   - Inside `build-nanogpt/edu_fineweb10B/` (current)
   - At `frontier-ready/data/` (shared)

3. **Package management:**
   - Keep separate `pyproject.toml` files
   - Use uv workspace
   - Flat dependency file at root

4. **Git submodule:**
   - Keep `build-nanogpt` as submodule
   - Merge into monorepo
   - Separate repo with git-subtree
