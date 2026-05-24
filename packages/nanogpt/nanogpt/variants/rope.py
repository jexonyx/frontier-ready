"""RoPE (Rotary Position Embeddings) variant.

Replaces learned positional embeddings (wpe) with rotary embeddings
applied to Q and K vectors in attention.

Usage example:
    from nanogpt.variants.rope import RotaryGPT
    from nanogpt.config import GPTConfig

    config = GPTConfig()
    model = RotaryGPT(config)
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..model import GPT, Block, CausalSelfAttention


class RotaryAttention(CausalSelfAttention):
    """CausalSelfAttention with rotary position embeddings on Q and K."""

    def __init__(self, config):
        super().__init__(config)
        # TODO: precompute rotary frequency table

    def forward(self, x):
        # TODO: apply rotary embeddings to Q and K before attention
        raise NotImplementedError("RoPE attention not yet implemented")


class RotaryBlock(Block):
    attn_cls = RotaryAttention


class RotaryGPT(GPT):
    """GPT model with RoPE positional embeddings instead of learned embeddings."""
    block_cls = RotaryBlock

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([self.block_cls(config) for _ in range(config.n_layer)]),
            ln_f=self.norm_cls(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        x = self.transformer.wte(idx)  # no position embedding addition
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
