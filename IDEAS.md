# Ideas

An unstructured collection of random thoughts that occur to me while going through this process. Nothing here is polished or necessarily even correct.

## Phase 1 — Architectural Modification Candidates

- **Differential Attention** (Microsoft, 2024): Computes attention as the difference of two softmax attention maps to reduce noise in attention weights. Clean single-variable swap in `CausalSelfAttention`. Genuine uncertainty about effect at 124M scale — worth investigating whether the noise-reduction benefit materialises at small model size or only emerges at larger scale.

- **Multi-head Latent Attention** (DeepSeek-V2): Compresses KV cache into a low-rank latent space, reducing memory and compute for KV projections. Modifies how attention is computed rather than which tokens are attended to, so information flow is preserved. Open question at 124M scale: does the compression hurt when the model is already small, or does the implicit regularisation help?

- **RoPE (Rotary Position Embeddings):** Replace learned positional embeddings (`wpe`) with rotary embeddings applied to Q and K vectors. The cleanest possible swap — remove the position embedding table and modify `CausalSelfAttention` to apply rotations. At 1024 context with 124M parameters, genuinely unclear whether RoPE's inductive bias beats learned embeddings that have enough data to learn the same patterns. Probably the single safest choice for Phase 1.

- **GQA (Grouped Query Attention):** Reduce KV heads from 12 to 4 (or 2 or 1) while keeping 12 query heads. Multiple query heads share the same key-value pair. At 124M the model is already small — does reducing KV capacity hurt disproportionately compared to larger models where GQA is standard? Slightly reduces parameter count, which needs accounting for in analysis.

- **SwiGLU Activation (replacing GELU in MLP):** Replace the MLP's `GELU(xW1)W2` with `SiLU(xW1) * (xV) · W2` — the gated activation used in LLaMA, PaLM, and most modern architectures. Changes the MLP block rather than attention. To keep parameter count constant, reduce hidden dimension by ~2/3 since SwiGLU adds a third projection matrix. Open question: does the gating benefit require scale, or does it appear at 124M?

- **Parallel Attention + MLP Blocks:** Instead of running attention then MLP sequentially in each block, run them in parallel and sum their outputs (used in PaLM, GPT-J). Same parameters, different computation graph. Unusually clean experiment — changes nothing about the components, only how they compose. Strong writeup angle: does the residual stream benefit from parallel updates, or does the sequential dependency (MLP seeing attention output) matter at small scale?

- **RMSNorm (replacing LayerNorm):** Replace all LayerNorm layers with RMSNorm, which drops the mean-centering step and only normalises by root-mean-square. Used in LLaMA and most post-2023 architectures. Simplest implementation of all candidates (~10 lines of code). Risk: the effect might be too small to produce an interesting loss curve difference at this scale, leaving a correct but thin writeup.
