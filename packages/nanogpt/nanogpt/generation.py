"""Text generation utilities for GPT models."""
import torch
import tiktoken
from torch.nn import functional as F


def generate(
    model,
    prompt,
    max_length=100,
    num_samples=1,
    top_k=50,
    temperature=1.0,
    device='cuda',
    seed=None
):
    """
    Generate text from a prompt using top-k sampling.

    Args:
        model: The GPT model to use for generation
        prompt: Text prompt to start generation
        max_length: Maximum length of generated text (in tokens)
        num_samples: Number of samples to generate
        top_k: Top-k for sampling (default: 50)
        temperature: Sampling temperature (higher = more random)
        device: Device to run on ('cuda' or 'cpu')
        seed: Random seed for reproducibility (optional)

    Returns:
        List of generated text strings
    """
    # Encode the prompt
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_samples, 1)  # (num_samples, T)
    x = tokens.to(device)

    # Create random generator for sampling
    sample_rng = torch.Generator(device=device)
    if seed is not None:
        sample_rng.manual_seed(seed)

    # Generate tokens
    model.eval()
    with torch.no_grad():
        while x.size(1) < max_length:
            # Forward pass
            logits, _ = model(x)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size) - take last position

            # Apply temperature
            logits = logits / temperature

            # Get probabilities
            probs = F.softmax(logits, dim=-1)

            # Top-k sampling
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)

            # Append to sequence
            x = torch.cat((x, xcol), dim=1)

    # Decode and return
    results = []
    for i in range(num_samples):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        results.append(decoded)

    return results
