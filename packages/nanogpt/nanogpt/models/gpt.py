"""GPT model implementation."""
import inspect
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..config.model_config import ModelConfig
from ..core.block import TransformerBlock
from ..core.embedding import (
    PositionEmbedding,
    LearnedPositionEmbedding,
    RoPEPositionEmbedding,
    NoPositionEmbedding,
)


class GPT(nn.Module):
    """GPT Language Model.

    A decoder-only transformer model following the GPT-2 architecture.
    Uses composition and dependency injection for modularity.
    """

    # Class attributes for customization (can be overridden in subclasses)
    block_cls = TransformerBlock
    norm_cls = nn.LayerNorm

    def __init__(self, config: ModelConfig, position_embedding: Optional[PositionEmbedding] = None):
        """Initialize GPT model.

        Args:
            config: Model configuration
            position_embedding: Custom position embedding (default: None)
                If None, creates position embedding based on config.position_embedding_type
        """
        super().__init__()
        self.config = config

        # Create position embedding if not provided
        if position_embedding is None:
            position_embedding = self._create_position_embedding(config)

        self.position_embedding = position_embedding

        # Transformer components
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([
                self.block_cls(config, layer_idx=i)
                for i in range(config.n_layer)
            ]),
            ln_f=self.norm_cls(config.n_embd),
        ))

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying scheme
        if config.tie_word_embeddings:
            self.transformer.wte.weight = self.lm_head.weight

        # Initialize parameters
        self.apply(self._init_weights)

    def _create_position_embedding(self, config: ModelConfig) -> PositionEmbedding:
        """Create position embedding based on config.

        Args:
            config: Model configuration

        Returns:
            Position embedding instance

        Raises:
            ValueError: If position_embedding_type is not recognized
        """
        if config.position_embedding_type == "learned":
            return LearnedPositionEmbedding(config.block_size, config.n_embd)
        elif config.position_embedding_type == "rope":
            return RoPEPositionEmbedding(
                config.block_size,
                config.n_embd,
                config.n_head,
            )
        elif config.position_embedding_type == "none":
            return NoPositionEmbedding()
        else:
            raise ValueError(
                f"Unknown position_embedding_type: {config.position_embedding_type}. "
                f"Options: 'learned', 'rope', 'none'"
            )

    def _init_weights(self, module):
        """Initialize model weights.

        Args:
            module: Module to initialize
        """
        # Import here to avoid circular dependency
        from ..core.initialization import ScaledLinear

        # ScaledLinear handles its own initialization via reset_parameters()
        if isinstance(module, ScaledLinear):
            pass  # Already initialized correctly
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initialization_config.std
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initialization_config.std
            )

    def forward(self, idx, targets=None):
        """Forward pass through the model.

        Args:
            idx: Input token indices of shape (B, T)
            targets: Target token indices of shape (B, T) (optional)

        Returns:
            Tuple of (logits, loss):
                - logits: Output logits of shape (B, T, vocab_size)
                - loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, "
            f"block size is only {self.config.block_size}"
        )

        # Token embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

        # Apply position embeddings
        x = self.position_embedding(tok_emb)

        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm and language model head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT-2 model weights from HuggingFace.

        Args:
            model_type: Model type ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")

        Returns:
            GPT model with pretrained weights
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024   # always 1024 for GPT model checkpoints

        # Create a from-scratch initialized model
        config = ModelConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # Initialize a HuggingFace/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type, verbose=True):
        """Configure AdamW optimizer with weight decay.

        Creates parameter groups for weight decay:
        - 2D parameters (weights) get weight decay
        - <2D parameters (biases, layernorms) don't get weight decay

        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            device_type: Device type ("cuda" or "cpu")
            verbose: Whether to print parameter counts

        Returns:
            Configured AdamW optimizer
        """
        # Start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        if verbose:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if verbose:
            print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer
